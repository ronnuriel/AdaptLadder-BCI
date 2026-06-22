const reveal = new IntersectionObserver((entries)=>{
  for (const entry of entries){
    if(entry.isIntersecting){ entry.target.classList.add('is-visible'); reveal.unobserve(entry.target); }
  }
},{threshold:.08});
for(const el of document.querySelectorAll('.section,.metric')){ reveal.observe(el); }

const triageConditions = {
  green: {
    condition: 'Recent previous state',
    action: 'Reuse previous input state',
    interpretation: 'Recent previous states were often near-native in the retrospective T15 analysis.',
    per: '9.97%',
    cost: '0 labels',
    note: '0–3 day previous-state reuse anchor.'
  },
  amber: {
    condition: 'Stale or geometrically unusual state',
    action: 'Retrieve older candidate using unlabeled geometry',
    interpretation: 'Geometry should retrieve candidates and warn about risk, not silently override recency.',
    per: '12.74%',
    cost: '0 labels',
    note: 'K=20 geometry-retrieval anchor.'
  },
  red: {
    condition: 'Stored state likely unreliable',
    action: 'Escalate to input-layer calibration',
    interpretation: 'When no stored state appears trustworthy, use labeled K trials to update only the input layer.',
    per: '18.38%',
    cost: 'K=20 labels',
    note: 'K=20 input-layer calibration anchor.'
  }
};

function classify(gap, drift){
  if (gap > 14 || drift > 0.76) return 'red';
  if (gap > 3 || drift > 0.52) return 'amber';
  return 'green';
}

function updateTriage(){
  const gapSlider = document.getElementById('gapSlider');
  const driftSlider = document.getElementById('driftSlider');
  if(!gapSlider || !driftSlider) return;
  const gap = Number(gapSlider.value);
  const drift = Number(driftSlider.value);
  const regime = classify(gap, drift);
  const state = triageConditions[regime];

  document.getElementById('gapValue').textContent = `${gap} ${gap === 1 ? 'day' : 'days'}`;
  document.getElementById('driftValue').innerHTML = `d<sub>cov</sub> = ${drift.toFixed(3)}`;

  const badge = document.getElementById('regimeBadge');
  badge.className = `regime-badge ${regime === 'green' ? '' : regime}`;
  badge.textContent = `${regime.toUpperCase()} REGIME ACTIVE`;
  document.getElementById('conditionText').textContent = state.condition;
  document.getElementById('actionText').textContent = state.action;
  document.getElementById('interpretationText').textContent = state.interpretation;
  document.getElementById('perPrimary').textContent = state.per;
  document.getElementById('perNote').textContent = state.note;
  document.getElementById('costText').textContent = state.cost;

  document.querySelectorAll('.flow-step').forEach(step => {
    step.classList.toggle('active', step.dataset.step === regime);
  });
}

for(const id of ['gapSlider','driftSlider']){
  const el = document.getElementById(id);
  if(el) el.addEventListener('input', updateTriage);
}
updateTriage();

const viewer = document.getElementById('figureViewer');
const caption = document.getElementById('figureCaption');
document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    if(viewer) viewer.src = btn.dataset.img;
    if(caption) caption.textContent = btn.dataset.caption;
  });
});
