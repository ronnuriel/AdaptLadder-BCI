(function(){
  const reveal = new IntersectionObserver((entries)=>{
    for (const entry of entries){
      if(entry.isIntersecting){ entry.target.classList.add('is-visible'); reveal.unobserve(entry.target); }
    }
  },{threshold:.08});
  for(const el of document.querySelectorAll('.section,.metric')) reveal.observe(el);

  const values = {
    native: 9.06,
    recentPrevious: 9.97,
    stalePrevious: 24.54,
    previousByK: {5:13.09, 10:12.84, 20:12.21},
    geometryByK: {5:14.00, 10:13.19, 20:12.74},
    inputCalByK: {5:21.10, 10:20.30, 20:18.38}
  };

  let selectedK = 20;

  const $ = (id) => document.getElementById(id);
  const pctWidth = (v) => `${Math.max(0, Math.min(100, (Number(v) / 30) * 100))}%`;
  const setBar = (id, labelId, label, value) => {
    const bar = $(id);
    const lab = $(labelId);
    if(!bar || !lab) return;
    lab.textContent = label;
    if(value === null || value === undefined){
      bar.style.setProperty('--w','0%');
      bar.textContent = '—';
      return;
    }
    bar.style.setProperty('--w', pctWidth(value));
    bar.textContent = Number(value).toFixed(2);
  };

  function classify(days){
    if(days <= 3) return 'green';
    if(days <= 14) return 'amber';
    return 'red';
  }

  function updateSimulator(){
    const daysSlider = $('daysSlider');
    if(!daysSlider) return;
    const days = Number(daysSlider.value);
    const regime = classify(days);
    const daysValue = $('daysValue');
    if(daysValue) daysValue.textContent = `${days} ${days === 1 ? 'day' : 'days'}`;

    const risk = days <= 3 ? 10 + (days/3)*20 : days <= 14 ? 35 + ((days-3)/11)*35 : 75 + Math.min(25, ((days-14)/31)*25);
    const riskFill = $('riskFill');
    if(riskFill) riskFill.style.width = `${risk.toFixed(0)}%`;
    const riskLabel = $('riskLabel');
    if(riskLabel) riskLabel.textContent = regime === 'green' ? 'Low' : regime === 'amber' ? 'Check morning K' : 'High / stale';

    const badge = $('regimeBadge');
    if(badge){
      badge.className = `regime-badge ${regime}`;
      badge.textContent = regime === 'green' ? 'GREEN · REUSE' : regime === 'amber' ? 'AMBER · CHECK K' : 'RED · ESCALATE';
    }

    if(regime === 'green'){
      $('decisionTitle').textContent = 'Reuse previous input state';
      $('decisionText').textContent = 'The previous state is recent. Start with zero-label reuse before asking for calibration data.';
      $('primaryLabel').textContent = 'Paper anchor';
      $('primaryPER').textContent = `${values.recentPrevious.toFixed(2)}%`;
      $('primaryNote').textContent = 'Observed previous-state PER for 0–3 day gap.';
      $('secondaryLabel').textContent = 'Calibration cost';
      $('secondaryPER').textContent = '0 labels';
      $('secondaryNote').textContent = 'No gradient update, no decoder change.';
      setBar('bar1','bar1Label','Recent previous', values.recentPrevious);
      setBar('bar2','bar2Label','Native reference', values.native);
      setBar('bar3','bar3Label','Fallback', null);
    } else if(regime === 'amber'){
      const prev = values.previousByK[selectedK];
      const geom = values.geometryByK[selectedK];
      $('decisionTitle').textContent = `Run morning probe with K = ${selectedK}`;
      $('decisionText').textContent = 'Use the morning probe to compare the previous state with retrieved candidates. Recency remains the default unless evidence suggests risk.';
      $('primaryLabel').textContent = 'Previous-state anchor';
      $('primaryPER').textContent = `${prev.toFixed(2)}%`;
      $('primaryNote').textContent = `Observed previous-state PER at K=${selectedK}.`;
      $('secondaryLabel').textContent = 'Geometry retrieval';
      $('secondaryPER').textContent = `${geom.toFixed(2)}%`;
      $('secondaryNote').textContent = 'Zero-label candidate retrieval; not automatic override.';
      setBar('bar1','bar1Label',`Previous, K=${selectedK}`, prev);
      setBar('bar2','bar2Label',`Geometry, K=${selectedK}`, geom);
      setBar('bar3','bar3Label','Native reference', values.native);
    } else {
      const cal = values.inputCalByK[selectedK];
      $('decisionTitle').textContent = 'Previous state is stale: retrieve or calibrate';
      $('decisionText').textContent = 'Beyond the stale threshold, previous reuse becomes risky. Try a stored candidate; if no state is reliable, calibrate the input layer only.';
      $('primaryLabel').textContent = 'Stale previous anchor';
      $('primaryPER').textContent = `${values.stalePrevious.toFixed(2)}%`;
      $('primaryNote').textContent = 'Observed previous-state PER when gap is >14 days.';
      $('secondaryLabel').textContent = `Input calibration K=${selectedK}`;
      $('secondaryPER').textContent = `${cal.toFixed(2)}%`;
      $('secondaryNote').textContent = 'Fallback when stored states appear unreliable.';
      setBar('bar1','bar1Label','Stale previous', values.stalePrevious);
      setBar('bar2','bar2Label',`Input calibration K=${selectedK}`, cal);
      setBar('bar3','bar3Label','Native reference', values.native);
    }
  }

  const daysSlider = $('daysSlider');
  if(daysSlider) daysSlider.addEventListener('input', updateSimulator);
  document.querySelectorAll('.k-button').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.k-button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      selectedK = Number(btn.dataset.k);
      updateSimulator();
    });
  });
  updateSimulator();

  const viewer = $('figureViewer');
  const caption = $('figureCaption');
  document.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      if(viewer) viewer.src = btn.dataset.img;
      if(caption) caption.textContent = btn.dataset.caption;
    });
  });
})();
