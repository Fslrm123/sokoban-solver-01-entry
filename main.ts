// sokoban-solver-01 entry: a push-optimal Sokoban solver for hallway-and-rooms levels.
//
// Strategy, in three layers:
//  1. Decompose. A square is "live" when a lone box on it can be pushed onto some goal (reverse pull
//     reachability from the goals). A box pushed onto a dead square can never reach a goal, so in every
//     solution each box stays inside its own connected component of live squares. Components holding
//     boxes are therefore independent rooms (the hallway squares are dead, so rooms never exchange
//     boxes), and the level's push optimum is the sum of the rooms' optima. Each room is solved on the
//     full board with the other rooms' boxes standing as walls; the player walks between rooms.
//  2. Per room, an A* search over pushes (a state is the boxes' squares plus the player's reachable
//     region named by its lowest square), so the first solution found takes the fewest pushes. The
//     heuristic is a minimum-cost bipartite matching of boxes to goals over precomputed lone-box push
//     distances (admissible: one push moves one box). Dead squares and frozen 2x2 blocks prune.
//  3. Fallbacks, all budgeted by expansions, never the clock (the run must be deterministic): a room
//     that exhausts its optimal budget is retried with weighted A*, then greedy best-first; a level
//     that does not decompose is searched whole. Every candidate answer is replay-verified before it
//     is written; an instance that defeats every search still gets its (empty) solution.txt.
//
// Local run: mkdir -p out && MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Directions in a fixed order (u, d, l, r): [dx, dy]; the letter a step writes, and a push writes.
const DIRS = [[0, -1], [0, 1], [-1, 0], [1, 0]];
const STEP_LETTER = ['u', 'd', 'l', 'r'];
const PUSH_LETTER = ['U', 'D', 'L', 'R'];
const INF = 1_000_000_000;

// Work budgets, in A* expansions (deterministic; the wall is never read).
const ROOM_BUDGET = 400_000; // per room, per attempt
const WHOLE_BUDGET = 1_500_000; // whole-level fallback, per attempt
const GLOBAL_CAP = 8_000_000; // across the whole set

/** A level in XSB text: width, height, walls (short rows and the outer frame are walls), goals, boxes, player. */
function parseLevel(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals = [];
  const boxes = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
  const goalSet = new Uint8Array(w * h);
  for (const g of goals) goalSet[g] = 1;
  return { w, h, wall, goals: goals.sort((a, b) => a - b), boxes: boxes.sort((a, b) => a - b), player, goalSet };
}

/**
 * Lone-box push distances to one goal: dist[x] is the fewest pushes that bring a box alone from x onto
 * the goal (the player teleporting to whichever stand square a push needs), by reverse pull BFS:
 * a box on x was pushed there from x-d only when x-d (the box's old square) and x-2d (the player's
 * stand square) are both floor. INF where no sequence of pushes reaches the goal.
 */
function pushDistancesTo(w, wall, goal) {
  const dist = new Int32Array(wall.length).fill(INF);
  const queue = new Int32Array(wall.length);
  let qh = 0;
  let qt = 0;
  dist[goal] = 0;
  queue[qt++] = goal;
  while (qh < qt) {
    const x = queue[qh++];
    const nd = dist[x] + 1;
    for (let k = 0; k < 4; k++) {
      const d = DIRS[k][1] * w + DIRS[k][0];
      const y = x - d;
      if (wall[y] === 0 && wall[y - d] === 0 && dist[y] > nd) {
        dist[y] = nd;
        queue[qt++] = y;
      }
    }
  }
  return dist;
}

/** Live squares: those from which a lone box can be pushed onto some goal. */
function liveSquares(w, wall, goals) {
  const live = new Uint8Array(wall.length);
  const queue = new Int32Array(wall.length);
  let qh = 0;
  let qt = 0;
  for (const g of goals) {
    live[g] = 1;
    queue[qt++] = g;
  }
  while (qh < qt) {
    const x = queue[qh++];
    for (let k = 0; k < 4; k++) {
      const d = DIRS[k][1] * w + DIRS[k][0];
      const y = x - d;
      if (wall[y] === 0 && wall[y - d] === 0 && !live[y]) {
        live[y] = 1;
        queue[qt++] = y;
      }
    }
  }
  return live;
}

/** Minimum-cost assignment of rows to distinct columns (Hungarian, O(n^2 m) for n rows <= m columns). */
function hungarian(cost) {
  const n = cost.length;
  const m = cost[0].length;
  const u = new Float64Array(n + 1);
  const v = new Float64Array(m + 1);
  const p = new Int32Array(m + 1); // p[j] = row (1-based) assigned to column j
  const way = new Int32Array(m + 1);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Float64Array(m + 1).fill(Infinity);
    const used = new Uint8Array(m + 1);
    do {
      used[j0] = 1;
      const i0 = p[j0];
      let delta = Infinity;
      let j1 = -1;
      for (let j = 1; j <= m; j++) {
        if (used[j]) continue;
        const cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (let j = 0; j <= m; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 !== 0);
  }
  let total = 0;
  for (let j = 1; j <= m; j++) if (p[j] !== 0) total += cost[p[j] - 1][j - 1];
  return total;
}

/** Minimum-cost matching by permutation (square matrices of at most 7): simple and obviously exact. */
function matchingByPermutation(cost) {
  const n = cost.length;
  const perm = Array.from({ length: n }, (_, i) => i);
  let best = INF;
  for (;;) {
    let s = 0;
    for (let i = 0; i < n; i++) {
      s += cost[i][perm[i]];
      if (s >= best) break;
    }
    if (s < best) best = s;
    // next permutation (lexicographic)
    let i = n - 2;
    while (i >= 0 && perm[i] >= perm[i + 1]) i--;
    if (i < 0) break;
    let j = n - 1;
    while (perm[j] <= perm[i]) j--;
    [perm[i], perm[j]] = [perm[j], perm[i]];
    for (let a = i + 1, b = n - 1; a < b; a++, b--) [perm[a], perm[b]] = [perm[b], perm[a]];
  }
  return best;
}

/**
 * The push search: A* over pushes for one (sub)level. A state is the boxes' squares (sorted) and the
 * player's reachable region named by its lowest square; g counts pushes, h is the matching bound.
 * weight 1 is push-optimal A*; weight > 1 is weighted A*; weight Infinity is greedy best-first.
 * Expands at most maxExpansions states (and never past the shared work cap): {pushes, moves} or null.
 */
function solvePush(level, maxExpansions, weight, work) {
  const { w, wall, goals, boxes: startBoxes, player: startPlayer, goalSet } = level;
  const size = wall.length;
  const nB = startBoxes.length;
  if (nB === 0) return { pushes: 0, moves: '' };
  if (nB > goals.length) return null;
  const solved = (bs) => bs.every((b) => goalSet[b]);
  if (solved(startBoxes)) return { pushes: 0, moves: '' };

  // Per-goal push distances and the live squares they imply.
  const goalDist = goals.map((g) => pushDistancesTo(w, wall, g));
  const live = new Uint8Array(size);
  for (let i = 0; i < size; i++) {
    if (wall[i]) continue;
    for (let gi = 0; gi < goals.length; gi++) {
      if (goalDist[gi][i] < INF) {
        live[i] = 1;
        break;
      }
    }
  }
  if (startBoxes.some((b) => !live[b])) return null;

  const square = nB === goals.length;
  const heuristic = (bs) => {
    const cost = bs.map((b) => goals.map((_, gi) => goalDist[gi][b]));
    const h = square && nB <= 7 ? matchingByPermutation(cost) : hungarian(cost);
    return h >= INF / 2 ? INF : h;
  };
  const h0 = heuristic(startBoxes);
  if (h0 >= INF) return null;

  const occupied = new Uint8Array(size);
  const regionSeen = new Int32Array(size); // the expanding state's player region
  const succSeen = new Int32Array(size); // a successor's player region (its canonical square)
  const stack = new Int32Array(size);
  let regionStamp = 0;
  let succStamp = 0;
  // Flood fill from `from` around walls and boxes, marking `seen` with `stamp`; returns the lowest square.
  const flood = (seen, stamp, from) => {
    let low = from;
    let n = 0;
    stack[n++] = from;
    seen[from] = stamp;
    while (n > 0) {
      const p = stack[--n];
      if (p < low) low = p;
      let q = p - w;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) { seen[q] = stamp; stack[n++] = q; }
      q = p + w;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) { seen[q] = stamp; stack[n++] = q; }
      q = p - 1;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) { seen[q] = stamp; stack[n++] = q; }
      q = p + 1;
      if (wall[q] === 0 && occupied[q] === 0 && seen[q] !== stamp) { seen[q] = stamp; stack[n++] = q; }
    }
    return low;
  };
  // Whether the box just pushed to t sits in a frozen 2x2 block of walls and boxes with a box off its goal.
  const frozenBlock = (t) => {
    for (const a of [t, t - 1, t - w, t - w - 1]) {
      const b = a + 1;
      const c = a + w;
      const d = c + 1;
      if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
      if ((occupied[a] && !goalSet[a]) || (occupied[b] && !goalSet[b]) || (occupied[c] && !goalSet[c]) || (occupied[d] && !goalSet[d])) return true;
    }
    return false;
  };
  const keyOf = size <= 65535 ? (bs, low) => String.fromCharCode(low, ...bs) : (bs, low) => `${low}|${bs.join(',')}`;

  // States, in creation order: boxes, the player's square (where the push left it), parent, g, and the
  // push that made the state (the box's old square and the direction), for move reconstruction.
  const stBoxes = [startBoxes.slice()];
  const stPlayer = [startPlayer];
  const stParent = [-1];
  const stG = [0];
  const stFrom = [-1];
  const stDir = [-1];
  const stKey = [''];
  const stF = [weight === Infinity ? h0 : weight * h0];
  const stSeq = [0];
  let seq = 1;
  const best = new Map();
  for (const b of startBoxes) occupied[b] = 1;
  stKey[0] = keyOf(startBoxes, flood(succSeen, ++succStamp, startPlayer));
  for (const b of startBoxes) occupied[b] = 0;
  best.set(stKey[0], 0);

  // Binary heap of state indices on (f asc, g desc, seq asc): deterministic.
  const heap = [0];
  const less = (a, b) => stF[a] < stF[b] || (stF[a] === stF[b] && (stG[a] > stG[b] || (stG[a] === stG[b] && stSeq[a] < stSeq[b])));
  const heapPush = (s) => {
    let i = heap.length;
    heap.push(s);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i], heap[p])) break;
      [heap[i], heap[p]] = [heap[p], heap[i]];
      i = p;
    }
  };
  const heapPop = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1], heap[c])) c++;
        if (!less(heap[c], heap[i])) break;
        [heap[i], heap[c]] = [heap[c], heap[i]];
        i = c;
      }
    }
    return top;
  };

  let expansions = 0;
  let found = -1;
  while (heap.length > 0) {
    const s = heapPop();
    const g = stG[s];
    if (best.get(stKey[s]) < g) continue; // a cheaper path to this state is already recorded
    const bs = stBoxes[s];
    if (solved(bs)) {
      found = s;
      break;
    }
    expansions++;
    if (expansions > maxExpansions || work.expansions + expansions > work.cap) {
      work.expansions += expansions;
      return null;
    }
    for (const b of bs) occupied[b] = 1;
    const here = ++regionStamp;
    flood(regionSeen, here, stPlayer[s]);
    for (let bi = 0; bi < bs.length; bi++) {
      const b = bs[bi];
      for (let di = 0; di < 4; di++) {
        const d = DIRS[di][1] * w + DIRS[di][0];
        const t = b + d;
        if (regionSeen[b - d] !== here || wall[t] !== 0 || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(t);
        let nb = null;
        let key = null;
        let h = INF;
        if (!frozen) {
          nb = bs.slice();
          nb[bi] = t;
          nb.sort((x, y) => x - y);
          key = keyOf(nb, flood(succSeen, ++succStamp, b));
          h = heuristic(nb);
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || h >= INF) continue;
        const ng = g + 1;
        const prev = best.get(key);
        if (prev !== undefined && prev <= ng) continue;
        best.set(key, ng);
        const idx = stBoxes.length;
        stBoxes.push(nb);
        stPlayer.push(b);
        stParent.push(s);
        stG.push(ng);
        stFrom.push(b);
        stDir.push(di);
        stKey.push(key);
        stF.push(weight === Infinity ? h : ng + weight * h);
        stSeq.push(seq++);
        heapPush(idx);
      }
    }
    for (const b of bs) occupied[b] = 0;
  }
  work.expansions += expansions;
  if (found < 0) return null;

  // Reconstruct the moves: walk to each push's stand square (shortest walk), then push.
  const path = [];
  for (let s = found; s > 0; s = stParent[s]) path.push(s);
  path.reverse();
  const prev = new Int32Array(size);
  const how = new Int8Array(size);
  const walkTo = (from, to) => {
    if (from === to) return '';
    prev.fill(-1);
    const queue = new Int32Array(size);
    let qh = 0;
    let qt = 0;
    queue[qt++] = from;
    prev[from] = from;
    while (qh < qt) {
      const p = queue[qh++];
      if (p === to) break;
      for (let di = 0; di < 4; di++) {
        const q = p + DIRS[di][1] * w + DIRS[di][0];
        if (wall[q] === 0 && occupied[q] === 0 && prev[q] === -1) {
          prev[q] = p;
          how[q] = di;
          queue[qt++] = q;
        }
      }
    }
    let s = '';
    for (let p = to; p !== from; p = prev[p]) s = STEP_LETTER[how[p]] + s;
    return s;
  };
  let moves = '';
  let player = startPlayer;
  let boxes = startBoxes;
  for (const s of path) {
    const from = stFrom[s];
    const di = stDir[s];
    const d = DIRS[di][1] * w + DIRS[di][0];
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += PUSH_LETTER[di];
    player = from;
    boxes = stBoxes[s];
  }
  return { pushes: path.length, moves };
}

/**
 * The level's rooms: the connected components of live squares that hold boxes. A box never leaves its
 * component in any solution (leaving would mean being pushed onto a dead square, from which no goal is
 * reachable), so each component is an independent subproblem. Null when a box starts on a dead square
 * or some component holds more or fewer boxes than goals: then the level does not decompose.
 */
function decompose(level) {
  const { w, wall, goals, boxes } = level;
  const size = wall.length;
  const live = liveSquares(w, wall, goals);
  if (boxes.some((b) => !live[b])) return null;
  const comp = new Int32Array(size).fill(-1);
  const rooms = [];
  for (let s = 0; s < size; s++) {
    if (!live[s] || comp[s] >= 0) continue;
    const cells = [s];
    comp[s] = rooms.length;
    for (let i = 0; i < cells.length; i++) {
      const p = cells[i];
      for (const q of [p - w, p + w, p - 1, p + 1]) {
        if (live[q] && comp[q] < 0) {
          comp[q] = rooms.length;
          cells.push(q);
        }
      }
    }
    rooms.push({ cells, boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[comp[b]].boxes.push(b);
  for (const g of goals) rooms[comp[g]].goals.push(g);
  const withBoxes = rooms.filter((r) => r.boxes.length > 0);
  if (withBoxes.length === 0 || withBoxes.some((r) => r.boxes.length !== r.goals.length)) return null;
  for (const r of withBoxes) {
    r.boxes.sort((a, b) => a - b);
    r.goals.sort((a, b) => a - b);
  }
  return withBoxes;
}

/** The square the player stands on after `moves` from `from` on a board `w` wide. */
function endOf(w, from, moves) {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

/**
 * The level solved room by room: each room's solution in turn (push-optimal first, then weighted, then
 * greedy), the other rooms' boxes standing as walls (on their goals once their room is done), the
 * player walking on from where the last room left it. Rooms are tried in order; a room the player
 * cannot yet solve is retried after the others. Null when some room defeats every attempt.
 */
function solveByRooms(level, rooms, work) {
  let player = level.player;
  let pushes = 0;
  let moves = '';
  const remaining = rooms.slice();
  while (remaining.length > 0) {
    let progressed = false;
    for (let i = 0; i < remaining.length; i++) {
      const room = remaining[i];
      const wall = level.wall.slice();
      for (const other of rooms) {
        if (other === room) continue;
        for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
      }
      const goalSet = new Uint8Array(level.wall.length);
      for (const g of room.goals) goalSet[g] = 1;
      const sub = { w: level.w, h: level.h, wall, goals: room.goals, boxes: room.boxes, player, goalSet };
      let solved = null;
      for (const weight of [1, 3, Infinity]) {
        solved = solvePush(sub, ROOM_BUDGET, weight, work);
        if (solved !== null) break;
        if (work.expansions >= work.cap) break;
      }
      if (solved === null) continue;
      pushes += solved.pushes;
      moves += solved.moves;
      player = endOf(level.w, player, solved.moves);
      room.done = true;
      remaining.splice(i, 1);
      progressed = true;
      break;
    }
    if (!progressed) return null;
  }
  return { pushes, moves };
}

/** Replay `moves` on the level: true when every step is legal and the replay ends with every box on a goal. */
function verify(level, moves) {
  const { w, wall, goalSet } = level;
  const box = new Uint8Array(wall.length);
  for (const b of level.boxes) box[b] = 1;
  let player = level.player;
  for (const ch of moves.toLowerCase()) {
    const d = ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
    const next = player + d;
    if (wall[next]) return false;
    if (box[next]) {
      const beyond = next + d;
      if (wall[beyond] || box[beyond]) return false;
      box[next] = 0;
      box[beyond] = 1;
    }
    player = next;
  }
  for (let i = 0; i < box.length; i++) if (box[i] && !goalSet[i]) return false;
  return true;
}

/** The moves solving one level: room by room when it decomposes, else a whole-level search. '' when nothing solves it. */
function solveLevel(text, work) {
  const level = parseLevel(text);
  if (level.boxes.length === 0 || level.player < 0) return '';
  if (level.boxes.every((b) => level.goalSet[b])) return '';
  const rooms = decompose(level);
  if (rooms !== null) {
    const byRooms = solveByRooms(level, rooms, work);
    if (byRooms !== null && verify(level, byRooms.moves)) return byRooms.moves;
  }
  for (const weight of [1, 3, Infinity]) {
    const whole = solvePush(level, WHOLE_BUDGET, weight, work);
    if (whole !== null && verify(level, whole.moves)) return whole.moves;
    if (work.expansions >= work.cap) break;
  }
  return '';
}

const work = { expansions: 0, cap: GLOBAL_CAP };
for (const id of readdirSync(I).sort()) {
  let moves = '';
  try {
    moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'), work);
  } catch {
    moves = '';
  }
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
