// Shared main-thread adapter: downloads are decoded in a disposable worker.
// Terminating it before GPU upload releases the large native decoder heap.
export function loadCompressedPly({artifact,baseURL,decoderPath='decoder/',onProgress=()=>{},
    createWorker=url=>new Worker(url,{type:'module'})}) {
    const base=new URL('.',baseURL);
    const worker=createWorker(new URL(decoderPath+'e19-decoder-worker.js',base));
    return new Promise((resolve,reject)=>{
        let complete=false;
        const finish=(error,value)=>{
            if(complete)return;
            complete=true;
            worker.terminate();
            if(error)reject(error);else resolve(value);
        };
        worker.onmessage=({data})=>{
            if(data.type==='progress')onProgress(data.message);
            else if(data.type==='error')finish(new Error(data.message));
            else if(data.type==='result'){
                if(!(data.ply instanceof ArrayBuffer)||data.report?.records!==artifact.count){
                    finish(new Error('Decoded artifact does not match its declared Gaussian population.'));
                }else finish(null,{ply:data.ply,report:data.report});
            }
        };
        worker.onerror=event=>finish(new Error(event.message||'Decoder worker failed.'));
        worker.postMessage({base:base.href,packagePath:artifact.packagePath,planPath:artifact.planPath,
                            decoderPath,imageBackend:'wasm',validate:false});
    });
}
