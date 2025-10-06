const mergeTweens = new Map();
const dissolveTweens = new Map();
const freezeTweens = new Map();
const baseDuration = 260;
const dissolveDuration = 110;
const freezeDuration = 120;

export function resetEffects() {
  mergeTweens.clear();
  dissolveTweens.clear();
  freezeTweens.clear();
}

export function scheduleMergeTween(id, startX, startY, radius) {
  mergeTweens.set(id, {
    start: performance.now(),
    startX,
    startY,
    radius,
    active: true
  });
}

export function scheduleDissolveTween(id, x, y, radius) {
  dissolveTweens.set(id, {
    start: performance.now(),
    x,
    y,
    radius
  });
}

export function sampleMergeScale(id) {
  const tween = mergeTweens.get(id);
  if (!tween) return 1;
  const elapsed = performance.now() - tween.start;
  const t = Math.min(1, elapsed / baseDuration);
  if (t >= 1) {
    mergeTweens.delete(id);
    return 1;
  }
  const overshoot = 1.08;
  const settle = 1 - Math.pow(1 - t, 2) * 0.12;
  return 1 + (overshoot - 1) * Math.sin(t * Math.PI) * 0.6 + (settle - 1) * t;
}

export function scheduleFreezeTween(id, startX, startY, radius, targetX, targetY) {
  freezeTweens.set(id, {
    start: performance.now(),
    startX,
    startY,
    radius,
    targetX,
    targetY
  });
}

export function sampleFreezeTransform(id) {
  const tween = freezeTweens.get(id);
  if (!tween) return null;
  const elapsed = performance.now() - tween.start;
  const t = Math.min(1, elapsed / freezeDuration);
  if (t >= 1) {
    freezeTweens.delete(id);
    return null;
  }
  const ease = t * t * (3 - 2 * t);
  const scale = 1 - ease * 0.6;
  const x = tween.startX + (tween.targetX - tween.startX) * ease;
  const y = tween.startY + (tween.targetY - tween.startY) * ease;
  return { x, y, scale };
}

export function renderScoreBurst(ctx, x, y, deltaScore) {
  const t = performance.now();
  ctx.save();
  ctx.translate(x, y - 36);
  ctx.fillStyle = 'rgba(255,255,255,0.9)';
  ctx.shadowColor = 'rgba(255, 150, 90, 0.55)';
  ctx.shadowBlur = 8;
  ctx.font = '20px "PingFang SC", "Segoe UI", sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(`+${deltaScore}`, 0, 0);
  ctx.restore();
}

export function renderDissolveHalo(ctx) {
  const now = performance.now();
  dissolveTweens.forEach((info, id) => {
    const elapsed = now - info.start;
    if (elapsed >= dissolveDuration) {
      dissolveTweens.delete(id);
      return;
    }
    const t = elapsed / dissolveDuration;
    const alpha = 0.4 * (1 - t);
    const radius = info.radius * (1 + 0.4 * t);
    ctx.save();
    ctx.strokeStyle = `rgba(255, 178, 125, ${alpha})`;
    ctx.lineWidth = 4 * (1 - t);
    ctx.beginPath();
    ctx.arc(info.x, info.y, radius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  });
}


export function renderDangerCountdown(ctx, ratio, width) {
  const height = 22;
  const y = 52;
  const barWidth = Math.max(0, Math.min(1, ratio)) * width;
  ctx.save();
  ctx.fillStyle = 'rgba(255, 86, 70, 0.25)';
  ctx.fillRect((width - barWidth) / 2, y - height / 2, barWidth, height);
  ctx.strokeStyle = 'rgba(255, 86, 70, 0.55)';
  ctx.lineWidth = 2;
  ctx.strokeRect(width * 0.15, y - height / 2, width * 0.7, height);
  ctx.restore();
}
