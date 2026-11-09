// Browser-side decoding of the existing E19 8+8 bitplane / tiled-SH package.
// WebAssembly decodes the V-PCC bitstream; the remaining operations use web APIs.
const f = Math.fround;
const progress = message => self.postMessage({type:'progress',message});
const sha = async bytes => [...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');
async function download(url) {
    const response=await fetch(url);
    if(!response.ok)throw new Error(`Download failed: ${url} (${response.status})`);
    return new Uint8Array(await response.arrayBuffer());
}
async function inflate(bytes) {
    return new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'))).arrayBuffer());
}
function morton(point,bits){let code=0;for(let bit=0;bit<bits;bit++)for(let axis=0;axis<3;axis++)code+=((point[axis]>>bit)&1)*2**(3*bit+axis);return code;}
function parseAnchors(text){
    const end=text.indexOf('end_header\n')+11;
    if(end<11)throw new Error('Missing anchor PLY header');
    const count=Number(text.slice(0,end).match(/element vertex (\d+)/)[1]);
    const points=text.slice(end).trim().split(/\r?\n/).map(line=>line.trim().split(/\s+/).slice(0,3).map(Number));
    if(points.length!==count)throw new Error('Anchor count mismatch');
    return points;
}
async function imagePixels(url, description, portableDecode) {
    const data=await download(url);
    if(description.encodedSha256 && await sha(data)!==description.encodedSha256)throw new Error('Encoded image integrity check failed');
    let w,h,rgba,offset=0,stride;
    if(portableDecode){
        const image=await portableDecode(data.buffer.slice(data.byteOffset,data.byteOffset+data.byteLength));
        w=image.width;h=image.height;rgba=image.data;stride=w*4;
    }else{
        if(typeof ImageDecoder==='undefined')throw new Error('Native ImageDecoder API unavailable. Use the portable wasm backend.');
        const decoder=new ImageDecoder({data,type:'image/avif',colorSpaceConversion:'none'});
        const {image}=await decoder.decode({frameIndex:0});
        w=image.displayWidth;h=image.displayHeight;
        rgba=new Uint8Array(image.allocationSize({format:'RGBA'}));
        const layout=await image.copyTo(rgba,{format:'RGBA'});
        offset=layout[0].offset;stride=layout[0].stride;
        image.close();decoder.close();
    }
    const channels=description.mode==='RGBA'?4:description.mode==='RGB'?3:1;
    const pixels=new Uint8Array(w*h*channels);
    for(let y=0;y<h;y++)for(let x=0;x<w;x++)for(let c=0;c<channels;c++)pixels[(y*w+x)*channels+c]=rgba[offset+y*stride+x*4+c];
    const hash=await sha(pixels);
    if(hash!==description.sha256)throw new Error(`Browser AVIF pixels differ from native: ${url.split('/').pop()} (${hash}).`);
    return {data:pixels,width:w,height:h,channels};
}
function dequantize(values, parameters, channels){
    let low=Infinity,high=-Infinity;
    for(const value of values){if(value<low)low=value;if(value>high)high=value;}
    const range=high-low||1;
    const output=new Float32Array(values.length);
    for(let i=0;i<values.length;i++){
        const c=parameters.min_values.length===1?0:i%channels;
        const min=f(parameters.min_values[c]), max=f(parameters.max_values[c]);
        const width=f(max-min)||1;
        output[i]=f(f(f((values[i]-low)/range)*width)+min);
    }
    return output;
}
function header(count){
    const names=['x','y','z','nx','ny','nz',...Array.from({length:3},(_,i)=>`f_dc_${i}`),...Array.from({length:45},(_,i)=>`f_rest_${i}`),'opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3'];
    return new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${count}\n${names.map(n=>`property float ${n}\n`).join('')}end_header\n`);
}
function comparePly(actual,golden,count,offset){
    const marker=new TextEncoder().encode('end_header\n');
    let goldenOffset=0;
    outer:for(let i=0;i<8192;i++){for(let j=0;j<marker.length;j++)if(golden[i+j]!==marker[j])continue outer;goldenOffset=i+marker.length;break;}
    const a=new DataView(actual.buffer,actual.byteOffset+offset),b=new DataView(golden.buffer,golden.byteOffset+goldenOffset);
    const groups={positions:[0,3],dc:[6,9],sh:[9,54],opacity:[54,55],scale:[55,58],rotation:[58,62]};
    const report={};
    for(const [name,[start,end]] of Object.entries(groups)){
        let max=0,squared=0,n=0;
        for(let i=0;i<count;i++)for(let k=start;k<end;k++){
            const delta=Math.abs(a.getFloat32((i*62+k)*4,true)-b.getFloat32((i*62+k)*4,true));
            if(!Number.isFinite(delta))throw new Error('Non-finite reconstructed attribute');
            max=Math.max(max,delta);squared+=delta*delta;n++;
        }
        report[name]={maxAbs:max,rmse:Math.sqrt(squared/n)};
    }
    return report;
}
self.onmessage=async({data:request})=>{
try{
    const start=performance.now(), base=new URL('.',request.base);
    const packageURL=new URL(request.packagePath||'./package/',base);
    const runtimeURL=new URL(request.decoderPath||'./',base);
    const [manifest,plan]=await Promise.all([
        fetch(new URL('manifest.json',packageURL)).then(r=>r.json()),
        fetch(new URL(request.planPath||'decode-plan.json',base)).then(r=>r.json())]);
    if(manifest.vpcc_geometry_bitdepth!==8||manifest.low_bits_layout!=='bitplanes')throw new Error('Unsupported prototype geometry profile');
    progress('Loading the WebAssembly V-PCC decoder…');
    const init=(await import(new URL('PccAppDecoder.mjs',runtimeURL))).default;
    const decoder=await init({noInitialRun:true,wasmBinary:await download(new URL('PccAppDecoder.wasm',runtimeURL)),print:()=>{},printErr:()=>{}});
    decoder.FS.writeFile('/stream.bin',await download(new URL('stream.bin',packageURL)));
    decoder.FS.writeFile('/inverse.cfg',await download(new URL('inverse.cfg',runtimeURL)));
    progress('Decoding the actual V-PCC stream in WebAssembly…');
    const t=performance.now();
    const code=decoder.callMain(['--compressedStreamPath=/stream.bin','--reconstructedDataPath=/anchors_%04d.ply',
        '--frameCount=1','--startFrameNumber=0','--inverseColorSpaceConversionConfig=/inverse.cfg','--nbThread=1','--computeChecksum=0']);
    if(code)throw new Error('TMC2 exited with status '+code);
    const points=parseAnchors(decoder.FS.readFile('/anchors_0000.ply',{encoding:'utf8'}));
    const decodeMs=performance.now()-t;
    if(points.length!==manifest.vpcc_anchors)throw new Error('Unexpected V-PCC anchor population');
    const lex=[...points].sort((a,b)=>a[0]-b[0]||a[1]-b[1]||a[2]-b[2]);
    const anchorBytes=new Uint8Array(lex.length*6),av=new DataView(anchorBytes.buffer);
    lex.forEach((p,i)=>p.forEach((v,k)=>av.setUint16(i*6+k*2,v,true)));
    const anchorHash=await sha(anchorBytes);
    if(plan.goldenAnchorsSha256 && anchorHash!==plan.goldenAnchorsSha256)throw new Error('WebAssembly anchor parity failed');
    points.sort((a,b)=>morton(a,8)-morton(b,8));
    progress('Reconstructing associations and exact fine-coordinate bits…');
    const association=await inflate(await download(new URL(manifest.association,packageURL)));
    if(association.length!==manifest.records*4)throw new Error('Invalid association size');
    const ids=new Uint32Array(manifest.records),deltas=new DataView(association.buffer,association.byteOffset);
    let id=0;
    for(let i=0;i<ids.length;i++){
        const packed=deltas.getUint32(i*4,true);
        id+=(packed>>>1)^-(packed&1);
        if(id<0||id>=points.length)throw new Error('Invalid anchor association');
        ids[i]=id;
    }
    const fine=new Uint16Array(manifest.records*3);
    for(let bit=0;bit<8;bit++){
        const plane=await inflate(await download(new URL(manifest.low_bitplanes[bit],packageURL)));
        if(plane.length!==Math.ceil(fine.length/8))throw new Error('Invalid residual plane size');
        for(let i=0;i<fine.length;i++)fine[i]|=((plane[i>>3]>>(i&7))&1)<<bit;
    }
    let portableDecode;
    if(request.imageBackend!=='native'){
        const avif=await import(new URL('vendor/avif/decode.js',runtimeURL));
        await avif.init({wasmBinary:await download(new URL('vendor/avif/codec/dec/avif_dec.wasm',runtimeURL))});
        portableDecode=avif.default;
    }
    progress('Decoding actual AVIF images and checking native pixel parity…');
    const images={};
    for(const [name,description] of Object.entries(plan.images))images[name]=await imagePixels(new URL('semantic_sog/'+name,packageURL),description,portableDecode);
    const q=plan.quantizers;
    const dc=dequantize(images['f_dc.avif'].data,q.f_dc,3);
    const opacity=dequantize(images['opacities.avif'].data,q.opacities,1);
    const scales=dequantize(images['scales.avif'].data,q.scales,3);
    const rotations=dequantize(images['quaternions.avif'].data,q.quaternions,4);
    const side=plan.codebookSide,K=side*side;
    const tiled=images['f_rest_centroids.avif'];
    const centroidRaw=new Uint8Array(K*45);
    for(let y=0;y<side;y++)for(let x=0;x<side;x++)for(let c=0;c<3;c++)for(let j=0;j<15;j++){
        centroidRaw[(y*side+x)*45+c*15+j]=tiled.data[((Math.floor(j/5)*side+y)*tiled.width+(j%5)*side+x)*3+c];
    }
    const centroids=dequantize(centroidRaw,q.f_rest_centroids,45);
    const labelsRaw=new Uint16Array(manifest.records);
    const lo=images['f_rest_labels_bytes_0.avif'].data,hi=images['f_rest_labels_bytes_1.avif'].data;
    for(let i=0;i<labelsRaw.length;i++)labelsRaw[i]=lo[i]|hi[i]<<8;
    const labels=dequantize(labelsRaw,q.f_rest_labels,1);
    progress('Assembling the reconstructed Gaussian scene…');
    const h=header(manifest.records),output=new Uint8Array(h.length+manifest.records*248);
    output.set(h);const view=new DataView(output.buffer,h.length);
    const transform=manifest.transform;
    for(let i=0;i<manifest.records;i++){
        const o=i*248,point=points[ids[i]];
        for(let axis=0;axis<3;axis++){
            const value=(point[axis]<<8)|fine[i*3+axis];
            const domain=value/transform.scale_to_quantized+transform.coordinate_domain_min;
            view.setFloat32(o+axis*4,Math.sign(domain)*Math.expm1(Math.abs(domain)),true);
            view.setFloat32(o+(6+axis)*4,dc[i*3+axis],true);
            view.setFloat32(o+(55+axis)*4,f(Math.log(Math.max(1e-7,f(Math.exp(scales[i*3+axis]))))),true);
        }
        const label=Math.trunc(labels[i]);
        if(label<0||label>=K)throw new Error('Invalid SH codebook index');
        for(let k=0;k<45;k++)view.setFloat32(o+(9+k)*4,centroids[label*45+k],true);
        // Match the native float32 sigmoid's intermediate rounding before
        // converting back to PLY logits; double-precision sigmoid drifts near 1.
        const alpha=Math.min(f(1-1e-7),Math.max(f(1e-7),f(1/f(1+f(Math.exp(-opacity[i]))))));
        view.setFloat32(o+54*4,f(Math.log(f(alpha/f(1-alpha)))),true);
        for(let k=0;k<4;k++)view.setFloat32(o+(58+k)*4,rotations[i*4+k],true);
    }
    const report={status:'decoded',records:manifest.records,anchors:points.length,anchorHash,
        wasmLinearMemoryBytes:decoder._emscripten_get_heap_size?.()??null,
        matchesNativeAnchors:plan.goldenAnchorsSha256?true:null,imagesMatchNative:true,
        imageDecoding:portableDecode?'portable AVIF WebAssembly; actual compressed images':'browser ImageDecoder',
        vpccMilliseconds:decodeMs,totalMilliseconds:performance.now()-start};
    if(request.validate){
        const golden=await download(new URL(request.goldenPath||'golden.ply',base));
        report.attributes=comparePly(output,golden,manifest.records,h.length);
        const tolerances={positions:1e-5,dc:1e-6,sh:1e-6,opacity:5e-4,scale:1e-6,rotation:1e-6};
        for(const [name,value] of Object.entries(report.attributes)){
            if(value.maxAbs>tolerances[name])throw new Error(`Native reconstruction parity failed for ${name}: ${value.maxAbs}`);
        }
        report.matchesNativeWithinTolerance=true;
    }
    self.postMessage({type:'result',report,ply:output.buffer},[output.buffer]);
}catch(error){self.postMessage({type:'error',message:error.stack||error.message});}
};
