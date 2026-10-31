import os
d=os.path.dirname(os.path.abspath(__file__)); src=os.path.join(d,'..','src')
s=open(os.path.join(d,'bench.src.html')).read()
for tag,f in (('/*INFER*/',os.path.join(src,'infer.js')),('/*LOADERS*/',os.path.join(src,'loaders.js')),('/*MODELS*/',os.path.join(d,'models.js')),('/*BENCH*/',os.path.join(d,'bench.js'))):
    s=s.replace(tag,open(f).read().replace('</script>','<\\/script>'))
open(os.path.join(d,'..','bench.html'),'w').write(s); print(len(s))
