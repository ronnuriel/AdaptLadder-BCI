function ready(fn){
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', fn);
  else fn();
}

ready(() => {
  const reveal = ('IntersectionObserver' in window) ? new IntersectionObserver((entries)=>{
    for (const entry of entries){
      if(entry.isIntersecting){ entry.target.classList.add('is-visible'); reveal.unobserve(entry.target); }
    }
  },{threshold:.08}) : null;
  document.querySelectorAll('.section,.metric').forEach(el => {
    if(reveal) reveal.observe(el); else el.classList.add('is-visible');
  });

  const triageConditions = {
    green: {
      condition: 'Recent previous state',
      action: 'Reuse previous input state',
      interpretation: 'The previous state is recent and geometry is not anomalous. Start with zero-shot reuse: no labels, no update.',
      per: '9.97%', cost: '0 labels', note: '0–3 day previous-state reuse anchor.', costNote: 'No gradient update.'
    },
    amber: {
      condition: 'Stale or geometrically unusual state',
      action: 'Retrieve older candidate using unlabeled geometry',
      interpretation: 'Use geometry as a risk sensor and candidate-retrieval signal. Do not blindly override recency without confidence.',
      per: '12.74%', cost: '0 labels', note: 'K=20 geometry-retrieval anchor.', costNote: 'Uses an unlabeled K-shot neural window.'
    },
    red: {
      condition: 'Stored state likely unreliable',
      action: 'Escalate to input-layer calibration',
      interpretation: 'When no stored state appears trustworthy, collect labeled K trials and update only the input layer.',
      per: '18.38%', cost: 'K=20 labels', note: 'K=20 input-layer calibration anchor.', costNote: 'GRU backbone and output head remain frozen.'
    }
  };

  function classify(gap, drift){
    if (gap > 14 || drift > 0.76) return 'red';
    if (gap > 3 || drift > 0.52) return 'amber';
    return 'green';
  }

  function setText(id, text){ const el = document.getElementById(id); if(el) el.textContent = text; }
  function setHTML(id, html){ const el = document.getElementById(id); if(el) el.innerHTML = html; }

  function updateTriage(){
    const gapSlider = document.getElementById('gapSlider');
    const driftSlider = document.getElementById('driftSlider');
    if(!gapSlider || !driftSlider) return;
    const gap = Number(gapSlider.value);
    const drift = Number(driftSlider.value);
    const regime = classify(gap, drift);
    const state = triageConditions[regime];

    setText('gapValue', `${gap} ${gap === 1 ? 'day' : 'days'}`);
    setHTML('driftValue', `d<sub>cov</sub> = ${drift.toFixed(3)}`);

    const badge = document.getElementById('regimeBadge');
    if(badge){
      badge.className = `regime-badge ${regime === 'green' ? '' : regime}`;
      badge.textContent = `${regime.toUpperCase()} REGIME ACTIVE`;
    }
    const panel = document.getElementById('outcomePanel');
    if(panel){ panel.className = `outcome-panel ${regime === 'green' ? '' : regime}`; }

    setText('conditionText', state.condition);
    setText('actionText', state.action);
    setText('interpretationText', state.interpretation);
    setText('perPrimary', state.per);
    setText('perNote', state.note);
    setText('costText', state.cost);
    setText('costNote', state.costNote);

    document.querySelectorAll('.flow-step').forEach(step => {
      step.classList.toggle('active', step.dataset.step === regime);
    });
  }

  ['gapSlider','driftSlider'].forEach(id => {
    const el = document.getElementById(id);
    if(el) el.addEventListener('input', updateTriage);
  });
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

  const video = document.getElementById('projectVideo');
  if(video){
    video.muted = true;
    video.defaultMuted = true;
    video.volume = 0;
  }
});
