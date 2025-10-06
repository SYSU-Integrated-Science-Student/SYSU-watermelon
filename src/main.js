import './style.css';
import { Engine, Composite, Bodies, Body, Events, Vector } from 'matter-js';
import sunLogo from '../images/Sun_Yat-sen_Univ_Logo.svg.png';
import scnuLogo from '../images/SouthChinaNormalUniv_Logo.svg.png';
import scauLogo from '../images/South_China_Agricultural_University_Logo.png';
import scutLogo from '../images/South_China_University_of_Technology_Logo_(Since_2022).svg.png';
import sustechLogo from '../images/Southern_University_of_Science_and_Technology_logo.svg.png';
import hkustgzLogo from '../images/HKUST.png';
import cuhkLogo from '../images/Emblem_of_CU.png';
import hitLogo from '../images/Harbin_Institute_of_Technology_logo.svg.png';
import njuLogo from '../images/NJU.svg.png';
import pkuLogo from '../images/Peking_University_seal.svg.png';
import {
  resetEffects,
  scheduleMergeTween,
  scheduleDissolveTween,
  scheduleFreezeTween,
  sampleMergeScale,
  renderScoreBurst,
  renderDangerCountdown,
  renderDissolveHalo,
  sampleFreezeTransform
} from './effects.js';

const canvas = document.getElementById('game');
const ctx = canvas.getContext('2d');
const scoreLabel = document.getElementById('score');
const overlay = document.getElementById('overlay');
const message = document.getElementById('message');
const restartBtn = document.getElementById('restart');
const gameContainer = document.querySelector('.game-container');
const rootStyle = document.documentElement.style;

const FRUIT_BASE = [
  { name: '华南农业大学', radius: 16, asset: scauLogo, color: '#43a047' },
  { name: '华南师范大学', radius: 20, asset: scnuLogo, color: '#1e88e5' },
  { name: '南方科技大学', radius: 24, asset: sustechLogo, color: '#ef6c00' },
  { name: '华南理工大学', radius: 28, asset: scutLogo, color: '#0d47a1' },
  { name: '哈尔滨工业大学', radius: 34, asset: hitLogo, color: '#0b5394' },
  { name: '港科大(广州)', radius: 42, asset: hkustgzLogo, color: '#1565c0' },
  { name: '香港中文大学', radius: 50, asset: cuhkLogo, color: '#6a1b9a' },
  { name: '南京大学', radius: 60, asset: njuLogo, color: '#7b1fa2' },
  { name: '北京大学', radius: 70, asset: pkuLogo, color: '#ad1457' },
  { name: '中山大学', radius: 84, asset: sunLogo, color: '#136f3b' }
];

const FRUIT_TYPES = FRUIT_BASE.map((type) => {
  const image = new Image();
  image.src = type.asset;
  return { ...type, image };
});

const INITIAL_SEQUENCE = [0, 0, 1, 2, 2, 3];
const RANDOM_POOL = [0, 1, 2, 3, 4, 5];

const DANGER_ZONE_HEIGHT = 60;
const DANGER_HOLD_MS = 1200;
const SPAWN_DELAY_MS = 240;
const MERGE_HOLD_MS = 80;
const MERGE_DISTANCE_RATIO = 1.05;
const MERGE_SPEED_THRESHOLD = 16;
const PHYSICS_STEP_MS = 1000 / 120;
const MAX_STEPS_PER_FRAME = 8;
const POINTER_SMOOTHING = 0.28;
const POINTER_MAX_SPEED = 18;
const LOCK_TIME_MS = 120;

const engine = Engine.create({ enableSleeping: false });
engine.gravity.y = 1;
engine.gravity.scale = 0.002;
const world = engine.world;

const fruitBodies = new Map();
const mergeCandidates = new Map();
const lockedBodies = new Map();

let spawnCounter = 0;
let currentFruitIndex = 0;
let upcomingFruitIndex = 0;
let score = 0;
let bestFruitReached = 0;
let gameOver = false;
let spawnReadyTime = 0;
let pointerX = canvas.width / 2;
let pointerTargetX = canvas.width / 2;
let lastFrameTime = null;
let dangerIndicator = 0;
let timeAccumulator = 0;
const scoreBursts = [];
let dangerHoldTime = 0;

function buildBounds() {
  const thickness = 200;
  const floor = Bodies.rectangle(
    canvas.width / 2,
    canvas.height + thickness / 2,
    canvas.width,
    thickness,
    { isStatic: true, restitution: 0.1, friction: 0.9 }
  );
  const leftWall = Bodies.rectangle(
    -thickness / 2,
    canvas.height / 2,
    thickness,
    canvas.height * 2,
    { isStatic: true, friction: 0.8 }
  );
  const rightWall = Bodies.rectangle(
    canvas.width + thickness / 2,
    canvas.height / 2,
    thickness,
    canvas.height * 2,
    { isStatic: true, friction: 0.8 }
  );
  const ceiling = Bodies.rectangle(
    canvas.width / 2,
    -thickness / 2,
    canvas.width,
    thickness,
    { isStatic: true, restitution: 0.1 }
  );

  Composite.add(world, [floor, leftWall, rightWall, ceiling]);
}

buildBounds();

function pairKey(a, b) {
  return a.id < b.id ? `${a.id}-${b.id}` : `${b.id}-${a.id}`;
}

function nextQueueIndex() {
  let index;
  if (spawnCounter < INITIAL_SEQUENCE.length) {
    index = INITIAL_SEQUENCE[spawnCounter];
  } else {
    const poolIndex = Math.floor(Math.random() * RANDOM_POOL.length);
    index = RANDOM_POOL[poolIndex];
  }
  spawnCounter += 1;
  return Math.min(index, FRUIT_TYPES.length - 1);
}

function isSpawnReady() {
  return !gameOver && performance.now() >= spawnReadyTime;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function bumpElement(element) {
  element.classList.remove('bump');
  // force reflow so animation can restart
  // eslint-disable-next-line no-unused-expressions
  element.offsetWidth;
  element.classList.add('bump');
}

function setDangerVisual(ratio) {
  const clamped = Math.max(0, Math.min(1, ratio));
  dangerIndicator = clamped;
  rootStyle.setProperty('--danger-ratio', clamped.toFixed(3));
  if (clamped > 0.001) {
    gameContainer.classList.add('danger');
  } else {
    gameContainer.classList.remove('danger');
  }
}

function createFruitBody(typeIndex, x, y) {
  const fruit = FRUIT_TYPES[typeIndex];
  const body = Bodies.circle(x, y, fruit.radius, {
    restitution: 0.2,
    friction: 0.18,
    frictionStatic: 0.4,
    frictionAir: 0.015,
    density: 0.0022
  });
  Body.setMass(body, fruit.radius * fruit.radius * 0.0032);
  body.fruit = true;

  const meta = { body, typeIndex, dangerTime: 0 };
  fruitBodies.set(body.id, meta);
  Composite.add(world, body);
  return meta;
}

function removeFruit(meta) {
  if (!meta || !fruitBodies.has(meta.body.id)) return;
  fruitBodies.delete(meta.body.id);
  Composite.remove(world, meta.body);
  cleanupMergeCandidates(meta.body.id);
}

function cleanupMergeCandidates(bodyId) {
  for (const [key, candidate] of mergeCandidates) {
    if (candidate.bodyA.id === bodyId || candidate.bodyB.id === bodyId) {
      mergeCandidates.delete(key);
    }
  }
}

function updateScoreDisplay(add = 0, reset = false) {
  if (reset) {
    score = 0;
  } else {
    score += add;
  }
  scoreLabel.textContent = `Score: ${score}`;
  bumpElement(scoreLabel);
}

function updateBestFruit(index) {
  if (index > bestFruitReached) {
    bestFruitReached = index;
  }
}

function setNextLabel() {
}

function showOverlay(text) {
  message.innerHTML = text;
  overlay.classList.add('active');
}

function hideOverlay() {
  overlay.classList.remove('active');
  message.textContent = '';
  setDangerVisual(0);
}

function triggerWin() {
  if (gameOver) return;
  updateScoreDisplay(100);
  showOverlay('Nice! 你合成出了大西瓜！<br>额外奖励 +100');
  gameOver = true;
}

function triggerLose(reason) {
  if (gameOver) return;
  setDangerVisual(1);
  showOverlay(reason);
  gameOver = true;
}

function resetGame() {
  for (const meta of fruitBodies.values()) {
    Composite.remove(world, meta.body);
  }
  fruitBodies.clear();
  mergeCandidates.clear();
  lockedBodies.forEach(({ body }) => Composite.remove(world, body));
  lockedBodies.clear();
  scoreBursts.length = 0;
  resetEffects();
  dangerHoldTime = 0;

  spawnCounter = 0;
  currentFruitIndex = nextQueueIndex();
  upcomingFruitIndex = nextQueueIndex();
  setNextLabel();
  updateScoreDisplay(0, true);
  bestFruitReached = 0;
  dangerIndicator = 0;
  pointerX = canvas.width / 2;
  pointerTargetX = pointerX;
  spawnReadyTime = performance.now();
  gameOver = false;
  timeAccumulator = 0;
  hideOverlay();
}

function dropPendingFruit() {
  if (!isSpawnReady()) return;
  const fruit = FRUIT_TYPES[currentFruitIndex];
  const spawnY = Math.max(fruit.radius + 16, 80);
  const meta = createFruitBody(currentFruitIndex, pointerX, spawnY);
  Body.setVelocity(meta.body, { x: 0, y: 12 });
  Body.setAngularVelocity(meta.body, (Math.random() - 0.5) * 0.25);

  currentFruitIndex = upcomingFruitIndex;
  upcomingFruitIndex = nextQueueIndex();
  setNextLabel();
  spawnReadyTime = performance.now() + SPAWN_DELAY_MS;
}

function considerCollisionPairs(event) {
  for (const pair of event.pairs) {
    const metaA = fruitBodies.get(pair.bodyA.id);
    const metaB = fruitBodies.get(pair.bodyB.id);
    if (!metaA || !metaB) continue;
    if (metaA.typeIndex !== metaB.typeIndex) continue;
    if (metaA.typeIndex >= FRUIT_TYPES.length - 1) continue;

    const key = pairKey(pair.bodyA, pair.bodyB);
    let candidate = mergeCandidates.get(key);
    if (!candidate) {
      candidate = { bodyA: pair.bodyA, bodyB: pair.bodyB, overlapTime: 0, active: false };
      mergeCandidates.set(key, candidate);
    }
    candidate.active = true;

    const dx = pair.bodyA.position.x - pair.bodyB.position.x;
    const dy = pair.bodyA.position.y - pair.bodyB.position.y;
    const dist = Math.hypot(dx, dy);
    const minDist = FRUIT_TYPES[metaA.typeIndex].radius + FRUIT_TYPES[metaB.typeIndex].radius;
    if (dist <= minDist * 1.05) {
      candidate.overlapTime = MERGE_HOLD_MS;
    }
  }
}

Events.on(engine, 'collisionStart', considerCollisionPairs);
Events.on(engine, 'collisionActive', considerCollisionPairs);
Events.on(engine, 'collisionEnd', (event) => {
  for (const pair of event.pairs) {
    mergeCandidates.delete(pairKey(pair.bodyA, pair.bodyB));
  }
});

function processMergeCandidates(deltaMs) {
  for (const [key, candidate] of mergeCandidates) {
    const metaA = fruitBodies.get(candidate.bodyA.id);
    const metaB = fruitBodies.get(candidate.bodyB.id);
    if (!metaA || !metaB) {
      mergeCandidates.delete(key);
      continue;
    }

    if (!candidate.active) {
      candidate.overlapTime = 0;
      continue;
    }
    candidate.active = false;

    const radiusA = FRUIT_TYPES[metaA.typeIndex].radius;
    const radiusB = FRUIT_TYPES[metaB.typeIndex].radius;
    const dx = candidate.bodyA.position.x - candidate.bodyB.position.x;
    const dy = candidate.bodyA.position.y - candidate.bodyB.position.y;
    const dist = Math.hypot(dx, dy);
    const minDist = radiusA + radiusB;
    const relative = Vector.sub(candidate.bodyA.velocity, candidate.bodyB.velocity);
    const relativeSpeed = Vector.magnitude(relative);

  const nearEnough = dist <= minDist * MERGE_DISTANCE_RATIO;
  const lowMotion =
    Math.abs(metaA.body.velocity.y) < 18 && Math.abs(metaB.body.velocity.y) < 18;
  const closeEnough = nearEnough || (dist <= minDist * 1.12 && lowMotion);
  const slowEnough =
    relativeSpeed < MERGE_SPEED_THRESHOLD || (lowMotion && relativeSpeed < MERGE_SPEED_THRESHOLD * 1.8);

    if (closeEnough && slowEnough) {
      candidate.overlapTime += deltaMs;
      if (candidate.overlapTime >= MERGE_HOLD_MS) {
        mergeCandidates.delete(key);
        performMerge(metaA, metaB);
      }
    } else {
      candidate.overlapTime = 0;
    }
  }
}

function performMerge(metaA, metaB) {
  if (!fruitBodies.has(metaA.body.id) || !fruitBodies.has(metaB.body.id)) return;
  const nextType = metaA.typeIndex + 1;
  const posX = (metaA.body.position.x + metaB.body.position.x) / 2;
  const posY = (metaA.body.position.y + metaB.body.position.y) / 2;
  const velX = (metaA.body.velocity.x + metaB.body.velocity.x) / 2;
  const velY = (metaA.body.velocity.y + metaB.body.velocity.y) / 2;

  const lockA = freezeFruit(metaA);
  const lockB = freezeFruit(metaB);
  scheduleDissolveTween(metaA.body.id, lockA.x, lockA.y, lockA.radius);
  scheduleDissolveTween(metaB.body.id, lockB.x, lockB.y, lockB.radius);
  scheduleFreezeTween(metaA.body.id, lockA.x, lockA.y, lockA.radius, posX, posY);
  scheduleFreezeTween(metaB.body.id, lockB.x, lockB.y, lockB.radius, posX, posY);

  const lift = FRUIT_TYPES[nextType].radius * 0.18;
  const newMeta = createFruitBody(nextType, posX, posY - lift);
  Body.setVelocity(newMeta.body, { x: velX * 0.4, y: velY * 0.3 - 7 });
  Body.applyForce(newMeta.body, newMeta.body.position, { x: 0, y: -newMeta.body.mass * 0.0025 });

  scheduleMergeTween(newMeta.body.id, posX, posY - lift, FRUIT_TYPES[nextType].radius);

  updateScoreDisplay(metaA.typeIndex + 1);
  updateBestFruit(nextType);

  scoreBursts.push({
    createdAt: performance.now(),
    x: posX,
    y: posY - lift,
    value: metaA.typeIndex + 1
  });

  if (nextType === FRUIT_TYPES.length - 1) {
    triggerWin();
  }
}

function freezeFruit(meta) {
  const radius = FRUIT_TYPES[meta.typeIndex].radius;
  const { x, y } = meta.body.position;
  const info = {
    body: meta.body,
    x,
    y,
    radius,
    typeIndex: meta.typeIndex,
    createdAt: performance.now()
  };
  Composite.remove(world, meta.body);
  fruitBodies.delete(meta.body.id);
  cleanupMergeCandidates(meta.body.id);
  lockedBodies.set(meta.body.id, info);
  return info;
}

function updateDanger(deltaMs) {
  let maxRatio = 0;
  let holdMs = 0;
  for (const meta of fruitBodies.values()) {
    const radius = FRUIT_TYPES[meta.typeIndex].radius;
    const topEdge = meta.body.position.y - radius;

    if (topEdge <= DANGER_ZONE_HEIGHT) {
      meta.dangerTime += deltaMs;
      const ratio = Math.min(1, meta.dangerTime / DANGER_HOLD_MS);
      maxRatio = Math.max(maxRatio, ratio);
      holdMs = Math.max(holdMs, meta.dangerTime);
      if (meta.dangerTime >= DANGER_HOLD_MS) {
        triggerLose('糟糕！校徽在顶部停留太久了。');
        break;
      }
    } else {
      meta.dangerTime = 0;
    }

    if (meta.body.position.y + radius < -radius * 0.5) {
      triggerLose('糟糕！校徽飞出边界了。');
      break;
    }
  }
  setDangerVisual(maxRatio);
  dangerHoldTime = holdMs;
}

function renderScene() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const topGradient = ctx.createLinearGradient(0, 0, 0, DANGER_ZONE_HEIGHT);
  const dangerAlpha = 0.15 + dangerIndicator * 0.3;
  topGradient.addColorStop(0, `rgba(255, 108, 85, ${dangerAlpha})`);
  topGradient.addColorStop(0.5, `rgba(255, 169, 113, ${dangerAlpha * 0.6})`);
  topGradient.addColorStop(1, dangerIndicator > 0 ? 'rgba(255, 198, 148, 0.12)' : 'rgba(0, 0, 0, 0.04)');
  ctx.fillStyle = topGradient;
  ctx.fillRect(0, 0, canvas.width, DANGER_ZONE_HEIGHT);
  ctx.strokeStyle = dangerIndicator > 0 ? 'rgba(255, 104, 82, 0.65)' : 'rgba(92, 109, 150, 0.45)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, DANGER_ZONE_HEIGHT);
  ctx.lineTo(canvas.width, DANGER_ZONE_HEIGHT);
  ctx.stroke();

  if (!gameOver && dangerIndicator > 0.05) {
    renderDangerCountdown(ctx, dangerIndicator, canvas.width);
    ctx.save();
    ctx.font = '14px "PingFang SC", "Segoe UI", sans-serif';
    ctx.fillStyle = 'rgba(255, 244, 240, 0.95)';
    ctx.textAlign = 'center';
    const remaining = Math.max(0, DANGER_HOLD_MS - dangerHoldTime);
    ctx.fillText(`危险！${(remaining / 1000).toFixed(1)}s`, canvas.width / 2, 60);
    ctx.restore();
  }

  ctx.fillStyle = 'rgba(0, 0, 0, 0.06)';
  ctx.fillRect(0, 120, canvas.width, 3);

  const renderFruitSprite = (img, x, y, radius, fallbackColor) => {
    if (img && img.complete) {
      const size = radius * 2;
      ctx.drawImage(img, x - size / 2, y - size / 2, size, size);
    } else {
      ctx.beginPath();
      ctx.fillStyle = fallbackColor;
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.55)';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.stroke();
  };

  for (const meta of fruitBodies.values()) {
    const fruit = FRUIT_TYPES[meta.typeIndex];
    const scale = sampleMergeScale(meta.body.id);
    const x = meta.body.position.x;
    const y = meta.body.position.y;
    const radius = fruit.radius * scale;
    renderFruitSprite(fruit.image, x, y, radius, fruit.color);
  }

  const now = performance.now();
  for (let i = scoreBursts.length - 1; i >= 0; i -= 1) {
    const burst = scoreBursts[i];
    const age = now - burst.createdAt;
    if (age > 600) {
      scoreBursts.splice(i, 1);
      continue;
    }
    const alpha = 1 - age / 600;
    ctx.save();
    ctx.globalAlpha = alpha;
    renderScoreBurst(ctx, burst.x, burst.y - age * 0.08, burst.value);
    ctx.restore();
  }

  renderDissolveHalo(ctx);

  lockedBodies.forEach((info, id) => {
    const age = now - info.createdAt;
    if (age > LOCK_TIME_MS) {
      lockedBodies.delete(id);
      return;
    }
    const progress = Math.min(1, age / LOCK_TIME_MS);
    const transform = sampleFreezeTransform(id) ?? {
      x: info.x,
      y: info.y,
      scale: Math.max(0.2, 1 - progress * 0.6)
    };
    const fruit = FRUIT_TYPES[info.typeIndex];
    const img = fruit.image;
    const radius = fruit.radius * transform.scale;
    ctx.save();
    ctx.globalAlpha = 0.75 * (1 - progress);
    if (img && img.complete) {
      const size = radius * 2;
      ctx.drawImage(img, transform.x - size / 2, transform.y - size / 2, size, size);
    } else {
      ctx.fillStyle = fruit.color;
      ctx.beginPath();
      ctx.arc(transform.x, transform.y, radius, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  });

  if (!gameOver && isSpawnReady()) {
    const fruit = FRUIT_TYPES[currentFruitIndex];
    const spawnY = Math.max(fruit.radius + 16, 80);
    ctx.save();
    ctx.globalAlpha = 0.65;
    renderFruitSprite(fruit.image, pointerX, spawnY, fruit.radius, fruit.color);
    ctx.restore();
  }
}

function updatePointerBounds() {
  const radius = FRUIT_TYPES[currentFruitIndex].radius;
  const margin = radius + 10;
  pointerTargetX = clamp(pointerTargetX, margin, canvas.width - margin);

  const delta = pointerTargetX - pointerX;
  const step = delta * POINTER_SMOOTHING;
  const limitedStep = Math.max(-POINTER_MAX_SPEED, Math.min(POINTER_MAX_SPEED, step));
  pointerX = clamp(pointerX + limitedStep, margin, canvas.width - margin);
}

function loop(timestamp) {
  if (!lastFrameTime) {
    lastFrameTime = timestamp;
  }
  const delta = Math.min(timestamp - lastFrameTime, 1000 / 20);
  timeAccumulator += delta;

  let steps = 0;
  let advancedMs = 0;
  while (timeAccumulator >= PHYSICS_STEP_MS && steps < MAX_STEPS_PER_FRAME) {
    Engine.update(engine, PHYSICS_STEP_MS);
    processMergeCandidates(PHYSICS_STEP_MS);
    advancedMs += PHYSICS_STEP_MS;
    timeAccumulator -= PHYSICS_STEP_MS;
    steps += 1;
  }

  const evalMs = advancedMs > 0 ? advancedMs : delta;

  if (!gameOver) {
    updateDanger(evalMs);
    updatePointerBounds();
  }

  renderScene();
  lastFrameTime = timestamp;
  requestAnimationFrame(loop);
}

requestAnimationFrame(loop);

canvas.addEventListener('mousemove', (event) => {
  if (gameOver) return;
  const rect = canvas.getBoundingClientRect();
  pointerTargetX = clamp(event.clientX - rect.left, 0, canvas.width);
});

canvas.addEventListener('click', () => {
  dropPendingFruit();
});

canvas.addEventListener('touchmove', (event) => {
  if (gameOver) return;
  const touch = event.touches[0];
  const rect = canvas.getBoundingClientRect();
  pointerTargetX = clamp(touch.clientX - rect.left, 0, canvas.width);
}, { passive: true });

canvas.addEventListener('touchend', () => {
  dropPendingFruit();
});

document.addEventListener('keydown', (event) => {
  if (event.code === 'Space') {
    event.preventDefault();
    dropPendingFruit();
  }
  if (event.code === 'KeyR') {
    resetGame();
  }
});

restartBtn.addEventListener('click', () => {
  resetGame();
});

resetGame();
