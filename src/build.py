s=open('index.src.html',encoding='utf-8').read()
for tag,f in (('/*I18N*/','i18n.js'),('/*ENUMS*/','enums.js'),('/*ANALYZER*/','analyzer.js'),('/*GRAPH*/','graph.js'),('/*INFER*/','infer.js'),('/*LOADERS*/','loaders.js'),('/*MODEL*/','model.js'),('/*DATA*/','data.js'),('/*PREGRAPHS*/','pregraphs.js'),('/*TEMPLATES*/','templates_tensors.js'),('/*TNEW*/','templates_new.js'),('/*APP*/','app.js')):
    s=s.replace(tag, open(f,encoding='utf-8').read().replace('</script','<\\/script'))
open('../index.html','w',encoding='utf-8').write(s); print(len(s))
