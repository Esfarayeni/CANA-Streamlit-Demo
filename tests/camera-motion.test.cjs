const test = require('node:test');
const assert = require('node:assert/strict');
const { CameraMotion } = require('../network_graph_component/camera-motion.js');
const overview = { yaw: -.65, pitch: .35, zoom: 1.8 };
const settle = camera => { for (let i = 0; i < 180; i++) camera.step(1 / 60); };

test('zoom is smooth, accumulates rapid clicks, and settles', () => {
  const camera = new CameraMotion(overview);
  camera.zoomBy(1.15); camera.zoomBy(1.15);
  assert.equal(camera.values.zoom, 1.8);
  camera.step(1 / 60);
  assert.ok(camera.values.zoom > 1.8 && camera.values.zoom < camera.zoom.target);
  settle(camera);
  assert.equal(camera.values.zoom, 1.8 * 1.15 * 1.15);
  assert.equal(camera.moving, false);
});

test('zoom reversal is continuous and respects both bounds', () => {
  const camera = new CameraMotion(overview);
  for (let i = 0; i < 40; i++) { camera.zoomBy(i % 2 ? .01 : 100); camera.step(1 / 60); assert.ok(camera.values.zoom >= 1 && camera.values.zoom <= 3.25); }
  camera.zoomBy(100); settle(camera); assert.equal(camera.values.zoom, 3.25);
  camera.zoomBy(.01); settle(camera); assert.equal(camera.values.zoom, 1);
});

test('physics gives the same trajectory at 60 Hz and 120 Hz', () => {
  const slow = new CameraMotion(overview), fast = new CameraMotion(overview);
  for (const camera of [slow, fast]) { camera.zoomBy(1.5); camera.release(2, -.5); }
  for (let i = 0; i < 24; i++) slow.step(1 / 60);
  for (let i = 0; i < 48; i++) fast.step(1 / 120);
  for (const key of ['yaw', 'pitch', 'zoom']) assert.ok(Math.abs(slow.values[key] - fast.values[key]) < .00001);
});

test('drag tracks directly and can interrupt release momentum', () => {
  const camera = new CameraMotion(overview);
  camera.rotateBy(.2, .1); assert.ok(Math.abs(camera.values.yaw - (overview.yaw + .2)) < .00001);
  camera.release(2, 1); camera.step(1 / 60);
  assert.ok(camera.values.yaw > overview.yaw + .2);
  camera.halt(); const stopped = camera.values; settle(camera);
  assert.deepEqual(camera.values, stopped);
});

test('release settles and pitch stays within safe limits', () => {
  const camera = new CameraMotion(overview); camera.release(4, 4); settle(camera);
  assert.equal(camera.moving, false);
  camera.rotateBy(0, 100); assert.equal(camera.values.pitch, 1.45);
  camera.release(0, 4); camera.step(1 / 60); assert.equal(camera.inertia.pitch, 0);
});

test('reset follows the shortest path and returns to the default zoom', () => {
  const camera = new CameraMotion({ yaw: overview.yaw + Math.PI * 8 + .4, pitch: -1, zoom: 3 });
  const start = camera.values.yaw;
  camera.reset(overview);
  assert.ok(Math.abs(camera.yaw.target - start) < Math.PI);
  assert.equal(camera.values.yaw, start);
  settle(camera);
  assert.ok(Math.abs(Math.sin(camera.values.yaw - overview.yaw)) < .00001);
  assert.equal(camera.values.zoom, overview.zoom);
  assert.equal(camera.moving, false);
});

test('reduced motion uses immediate changes and no release inertia', () => {
  const camera = new CameraMotion(overview);
  camera.zoomBy(1.2, true); assert.equal(camera.values.zoom, 2.16);
  camera.release(4, 4, true); assert.equal(camera.moving, false);
  camera.rotateBy(1, -.5); camera.reset(overview, true);
  assert.ok(Math.abs(camera.values.yaw - overview.yaw) < .00001);
  assert.equal(camera.values.zoom, overview.zoom);
  assert.equal(camera.moving, false);
});
