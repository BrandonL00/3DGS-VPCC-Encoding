import * as pc from './vendor/playcanvas-2.23.3.mjs';

const $ = id => document.getElementById(id);
const canvas = $('canvas');
const stage = $('stage');
const loading = $('loading');
const selector = $('budget');
const mb = bytes => `${(bytes / 1e6).toFixed(2)} MB`;
let app, camera, manifest, activeAsset, activeEntity, activeURL;
let target, worldUp;

function status(title, detail, progress = 0) {
    loading.hidden = false;
    $('load-title').textContent = title;
    $('load-detail').textContent = detail;
    $('progress').value = progress;
}

function resize() {
    if (app) app.resizeCanvas(stage.clientWidth, stage.clientHeight);
}

function resetCamera() {
    const c = manifest.camera;
    target = new pc.Vec3(...c.target);
    worldUp = new pc.Vec3(...c.up).normalize();
    camera.setPosition(...c.position);
    camera.lookAt(target, worldUp);
}

function orbit(dx, dy) {
    const offset = camera.getPosition().clone().sub(target);
    const yaw = new pc.Quat().setFromAxisAngle(worldUp, -dx * 0.25);
    yaw.transformVector(offset, offset);
    const right = new pc.Vec3().cross(worldUp, offset).normalize();
    const pitch = new pc.Quat().setFromAxisAngle(right, -dy * 0.25);
    const next = pitch.transformVector(offset, new pc.Vec3());
    if (Math.abs(next.clone().normalize().dot(worldUp)) < 0.995) offset.copy(next);
    camera.setPosition(target.clone().add(offset));
    camera.lookAt(target, worldUp);
}

function pan(dx, dy) {
    const distance = camera.getPosition().clone().sub(target).length();
    const scale = 2 * distance * Math.tan(camera.camera.fov * Math.PI / 360) / stage.clientHeight;
    const delta = camera.right.clone().mulScalar(-dx * scale).add(camera.up.clone().mulScalar(dy * scale));
    target.add(delta);
    camera.setPosition(camera.getPosition().clone().add(delta));
}

function zoom(factor) {
    const offset = camera.getPosition().clone().sub(target);
    const distance = Math.min(100, Math.max(0.08, offset.length() * factor));
    camera.setPosition(target.clone().add(offset.normalize().mulScalar(distance)));
}

function controls() {
    const pointers = new Map();
    canvas.addEventListener('contextmenu', event => event.preventDefault());
    canvas.addEventListener('pointerdown', event => {
        canvas.focus();
        canvas.setPointerCapture(event.pointerId);
        pointers.set(event.pointerId, {x: event.clientX, y: event.clientY, pan: event.button !== 0 || event.shiftKey});
    });
    canvas.addEventListener('pointermove', event => {
        const previous = pointers.get(event.pointerId);
        if (!previous) return;
        const dx = event.clientX - previous.x, dy = event.clientY - previous.y;
        const other = [...pointers.entries()].find(([id]) => id !== event.pointerId)?.[1];
        if (other) {
            const before = Math.hypot(previous.x - other.x, previous.y - other.y);
            const after = Math.hypot(event.clientX - other.x, event.clientY - other.y);
            if (before > 1 && after > 1) zoom(before / after);
            pan(dx / 2, dy / 2);
        } else if (previous.pan) pan(dx, dy);
        else orbit(dx, dy);
        previous.x = event.clientX;
        previous.y = event.clientY;
    });
    for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) {
        canvas.addEventListener(type, event => pointers.delete(event.pointerId));
    }
    canvas.addEventListener('wheel', event => {
        event.preventDefault();
        zoom(Math.exp(Math.max(-100, Math.min(100, event.deltaY)) * 0.002));
    }, {passive: false});
    canvas.addEventListener('keydown', event => {
        if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', '+', '-', '=', 'r', 'R'].includes(event.key)) event.preventDefault();
        if (event.key === 'ArrowLeft') orbit(-20, 0);
        if (event.key === 'ArrowRight') orbit(20, 0);
        if (event.key === 'ArrowUp') orbit(0, -20);
        if (event.key === 'ArrowDown') orbit(0, 20);
        if (event.key === '+' || event.key === '=') zoom(0.9);
        if (event.key === '-') zoom(1.1);
        if (event.key.toLowerCase() === 'r') resetCamera();
    });
}

async function fetchPart(part, onProgress) {
    const response = await fetch(new URL(part.url, import.meta.url));
    if (!response.ok) throw new Error(`Artifact download failed (${response.status}).`);
    const reader = response.body.getReader();
    const bytes = new Uint8Array(part.bytes);
    let offset = 0;
    while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        if (offset + value.length > bytes.length) throw new Error('Unexpected artifact size.');
        bytes.set(value, offset);
        offset += value.length;
        onProgress(value.length);
    }
    if (offset !== part.bytes) throw new Error('Artifact download was incomplete.');
    const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(x => x.toString(16).padStart(2, '0')).join('');
    if (hash !== part.sha256) throw new Error('Artifact integrity check failed.');
    return bytes;
}

function releaseCurrent() {
    activeEntity?.destroy();
    if (activeAsset) {
        activeAsset.unload();
        app.assets.remove(activeAsset);
    }
    if (activeURL) URL.revokeObjectURL(activeURL);
    activeEntity = activeAsset = activeURL = undefined;
    $('download').removeAttribute('href');
    $('download').setAttribute('aria-disabled', 'true');
}

async function loadArtifact(id) {
    const artifact = manifest.artifacts.find(item => item.id === id);
    selector.disabled = true;
    loading.classList.remove('error');
    releaseCurrent();
    $('poster').hidden = false;
    $('active-label').textContent = artifact.label;
    $('count').textContent = artifact.count.toLocaleString();
    $('encoded').textContent = mb(artifact.encodedBytes);
    $('ply-size').textContent = mb(artifact.plyBytes);
    $('quality').textContent = `${artifact.sourcePsnr.toFixed(2)} dB / ${artifact.sourceVmaf.toFixed(2)}`;
    let received = 0;
    status(`Loading ${artifact.label}`, `0 / ${mb(artifact.plyBytes)}`);
    try {
        const pieces = [];
        for (const part of artifact.chunks) {
            pieces.push(await fetchPart(part, amount => {
                received += amount;
                status(`Loading ${artifact.label}`, `${mb(received)} / ${mb(artifact.plyBytes)}`, received / artifact.plyBytes);
            }));
        }
        const blob = new Blob(pieces, {type: 'application/octet-stream'});
        activeURL = URL.createObjectURL(blob);
        status('Preparing scene', 'Uploading Gaussian data to your GPU…', 1);
        activeAsset = new pc.Asset(`Bicycle ${id}`, 'gsplat', {url: activeURL, filename: `bicycle-${id}.ply`});
        app.assets.add(activeAsset);
        await new Promise((resolve, reject) => {
            activeAsset.once('load', resolve);
            activeAsset.once('error', error => reject(new Error(String(error))));
            app.assets.load(activeAsset);
        });
        activeEntity = new pc.Entity('Bicycle reconstruction');
        activeEntity.setEulerAngles(0, 0, 180);
        activeEntity.addComponent('gsplat', {asset: activeAsset});
        app.root.addChild(activeEntity);
        $('download').href = activeURL;
        $('download').download = `bicycle-C18.3-E19-${id}.ply`;
        $('download').setAttribute('aria-disabled', 'false');
        // Wait for the first render, not just the asset fetch, before removing the poster.
        app.once('frameend', () => {
            loading.hidden = true;
            $('poster').hidden = true;
        });
        document.title = `Bicycle ${artifact.label} · E19`;
        history.replaceState(null, '', `#${id}`);
    } catch (error) {
        releaseCurrent();
        loading.classList.add('error');
        status('Could not load the scene', `${error.message}\nTry a smaller budget or reload the page.`);
    } finally {
        selector.disabled = false;
    }
}

async function main() {
    const response = await fetch('./artifacts.json');
    if (!response.ok) throw new Error('Artifact manifest is unavailable.');
    manifest = await response.json();
    app = new pc.Application(canvas, {graphicsDeviceOptions: {antialias: false, alpha: false}});
    app.setCanvasFillMode(pc.FILLMODE_NONE, stage.clientWidth, stage.clientHeight);
    app.setCanvasResolution(pc.RESOLUTION_AUTO);
    app.graphicsDevice.maxPixelRatio = Math.min(window.devicePixelRatio, 1.5);
    app.scene.toneMapping = pc.TONEMAP_NONE;
    camera = new pc.Entity('Camera');
    camera.addComponent('camera', {clearColor: new pc.Color(0.07, 0.09, 0.075),
        fov: manifest.camera.fov, nearClip: 0.01, farClip: 1000});
    app.root.addChild(camera);
    resetCamera();
    controls();
    new ResizeObserver(resize).observe(stage);
    resize();
    app.start();
    selector.replaceChildren(...manifest.artifacts.map(item => new Option(item.label, item.id)));
    const initial = manifest.artifacts.some(item => item.id === location.hash.slice(1)) ? location.hash.slice(1) : '64k';
    selector.value = initial;
    selector.addEventListener('change', () => loadArtifact(selector.value));
    $('reset').addEventListener('click', resetCamera);
    $('fullscreen').addEventListener('click', async () => {
        try {
            if (document.fullscreenElement) await document.exitFullscreen();
            else await stage.requestFullscreen();
        } catch (error) { console.warn('Fullscreen unavailable:', error.message); }
    });
    canvas.addEventListener('webglcontextlost', event => {
        event.preventDefault();
        loading.classList.add('error');
        status('Graphics memory unavailable', 'Reload the page and try the 64k scene.');
    });
    await loadArtifact(initial);
}

main().catch(error => {
    loading.classList.add('error');
    status('Viewer unavailable', `${error.message}\nUse a browser with WebGL2 graphics enabled.`);
});
