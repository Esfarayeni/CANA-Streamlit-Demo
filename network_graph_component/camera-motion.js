/* Dependency-free camera physics shared by the renderer and tests. */
(function (root) {
  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
  const axis = value => ({ value, target: value, velocity: 0 });
  function spring(state, seconds) {
    // Exact critically damped solution, stable across display refresh rates.
    const frequency = 24, offset = state.value - state.target;
    const c = state.velocity + frequency * offset, decay = Math.exp(-frequency * seconds);
    state.value = state.target + (offset + c * seconds) * decay;
    state.velocity = (state.velocity - frequency * c * seconds) * decay;
    if (Math.abs(state.value - state.target) < .0001 && Math.abs(state.velocity) < .001) {
      state.value = state.target; state.velocity = 0;
    }
  }
  class CameraMotion {
    constructor(initial) {
      this.yaw = axis(initial.yaw); this.pitch = axis(clamp(initial.pitch, -1.45, 1.45));
      this.zoom = axis(clamp(initial.zoom, 1, 3.25));
      this.inertia = { yaw: 0, pitch: 0 }; this.resetting = false;
    }
    get values() { return { yaw: this.yaw.value, pitch: this.pitch.value, zoom: this.zoom.value }; }
    get moving() { return this.resetting || this.zoom.value !== this.zoom.target || this.zoom.velocity !== 0 || this.inertia.yaw !== 0 || this.inertia.pitch !== 0; }
    zoomBy(factor, reduced = false) {
      this.zoom.target = clamp(this.zoom.target * factor, 1, 3.25);
      if (reduced) { this.zoom.value = this.zoom.target; this.zoom.velocity = 0; }
    }
    halt() {
      this.resetting = false; this.inertia = { yaw: 0, pitch: 0 };
      for (const state of [this.yaw, this.pitch]) { state.target = state.value; state.velocity = 0; }
    }
    rotateBy(yaw, pitch) {
      this.yaw.value += yaw; this.pitch.value = clamp(this.pitch.value + pitch, -1.45, 1.45);
      this.yaw.target = this.yaw.value; this.pitch.target = this.pitch.value;
    }
    release(yawVelocity, pitchVelocity, reduced = false) {
      this.inertia = reduced ? { yaw: 0, pitch: 0 } : { yaw: clamp(yawVelocity, -4, 4), pitch: clamp(pitchVelocity, -4, 4) };
    }
    reset(view, reduced = false) {
      this.halt();
      // Reset by the shortest angular path, even after several full rotations.
      const distance = Math.atan2(Math.sin(view.yaw - this.yaw.value), Math.cos(view.yaw - this.yaw.value));
      this.yaw.target = this.yaw.value + distance; this.pitch.target = clamp(view.pitch, -1.45, 1.45);
      this.zoom.target = clamp(view.zoom, 1, 3.25); this.resetting = true;
      if (reduced) this.finish();
    }
    finish() {
      for (const state of [this.yaw, this.pitch, this.zoom]) { state.value = state.target; state.velocity = 0; }
      this.halt();
    }
    step(seconds) {
      seconds = clamp(seconds, 0, .05); spring(this.zoom, seconds);
      if (this.zoom.value < 1 || this.zoom.value > 3.25) {
        this.zoom.value = clamp(this.zoom.value, 1, 3.25); this.zoom.velocity = 0;
      }
      if (this.resetting) {
        spring(this.yaw, seconds); spring(this.pitch, seconds);
        if ([this.yaw, this.pitch].every(state => state.value === state.target && state.velocity === 0)) this.resetting = false;
      } else {
        const decay = Math.exp(-7 * seconds);
        this.rotateBy(this.inertia.yaw * (1 - decay) / 7, this.inertia.pitch * (1 - decay) / 7);
        for (const key of ["yaw", "pitch"]) {
          this.inertia[key] *= decay;
          if (Math.abs(this.inertia[key]) < .005) this.inertia[key] = 0;
        }
        if (Math.abs(this.pitch.value) >= 1.45) this.inertia.pitch = 0;
      }
    }
  }
  if (typeof module !== "undefined" && module.exports) module.exports = { CameraMotion };
  else root.CanaCameraMotion = CameraMotion;
})(typeof window !== "undefined" ? window : globalThis);
