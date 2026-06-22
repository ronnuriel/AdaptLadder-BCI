const reveal = new IntersectionObserver((entries)=>{
  for (const entry of entries){
    if(entry.isIntersecting){ entry.target.classList.add('is-visible'); reveal.unobserve(entry.target); }
  }
},{threshold:.08});
for(const el of document.querySelectorAll('.section,.metric')){ reveal.observe(el); }

const style = document.createElement('style');
style.textContent = `.section,.metric{opacity:0;transform:translateY(18px);transition:opacity .6s ease, transform .6s ease}.section.is-visible,.metric.is-visible{opacity:1;transform:none}`;
document.head.appendChild(style);
