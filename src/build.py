"""Build the site: index.html (Thai) and en/index.html (English) from index.src.html and the src/*.js modules.

Each page gets its own language, title, description, canonical URL and hreflang links, and every element marked
with data-i / data-ih / data-iph is filled with its text at build time, so the HTML that search engines fetch
already holds the words a visitor sees. The scripts still re-render the text at runtime.
"""
import html, json, os, re, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://joessattes.github.io/ai-model-explorer/"
os.chdir(HERE)

src = open("index.src.html", encoding="utf-8").read()
for tag, f in (("/*I18N*/", "i18n.js"), ("/*ENUMS*/", "enums.js"), ("/*ANALYZER*/", "analyzer.js"), ("/*GRAPH*/", "graph.js"),
               ("/*INFER*/", "infer.js"), ("/*LOADERS*/", "loaders.js"), ("/*MODEL*/", "model.js"), ("/*DATA*/", "data.js"),
               ("/*PREGRAPHS*/", "pregraphs.js"), ("/*TEMPLATES*/", "templates_tensors.js"), ("/*TNEW*/", "templates_new.js"), ("/*APP*/", "app.js")):
    src = src.replace(tag, open(f, encoding="utf-8").read().replace("</script", "<\\/script"))

# UI strings, read from i18n.js with Node so there is one source of truth
I18N = json.loads(subprocess.run(["node", "-e", "global.window={};require('./i18n.js');process.stdout.write(JSON.stringify(window.I18N))"],
                                 capture_output=True, text=True, check=True).stdout)

PAGES = {
    "th": dict(out="../index.html", root="", canon=BASE, alt="en/", ogloc="th_TH", ogalt="en_US",
               title="AI Model Explorer | ดูโครงสร้างโมเดลบน Hugging Face ไม่ต้องโหลด",
               desc="วางชื่อ repo บน Hugging Face แล้วเห็นโครงสร้างโมเดลแยกเป็นส่วน: layer, attention, MoE, SSM, จำนวน parameter และ precision อ่านแค่ header ของ safetensors ไม่โหลด weight ใช้ฟรี ภาษาไทยและอังกฤษ"),
    "en": dict(out="../en/index.html", root="../", canon=BASE + "en/", alt="../", ogloc="en_US", ogalt="th_TH",
               title="AI Model Explorer | See Inside Any Hugging Face Model",
               desc="Paste a Hugging Face repo and see its architecture split into parts: layers, attention, MoE, SSM, parameter counts and precision (FP8, NVFP4, GPTQ). Reads only the safetensors header, never the weights. Free, open source."),
}

def text(key, lang):
    e = I18N.get(key)
    return e[1 if lang == "en" else 0] if e else None

def fill(body, lang):
    """Put each data-i / data-ih string into its element; data-iph into the placeholder. Scripts are left alone."""
    def tag_text(m):
        v = text(m.group(3), lang)
        if v is None: return m.group(0)
        return m.group(1) + (v if m.group(2) == "ih" else html.escape(v, quote=False)) + m.group(5)
    def placeholder(m):
        v = text(m.group(2), lang)
        return m.group(0) if v is None else m.group(1) + html.escape(v, quote=True) + m.group(3)
    parts = re.split(r"(<script\b.*?</script>)", body, flags=re.S)
    for k in range(0, len(parts), 2):
        t = re.sub(r'(<[a-zA-Z][^<>]*?\sdata-(i|ih)="([^"]+)"[^<>]*>)([^<]*)(</)', tag_text, parts[k])
        parts[k] = re.sub(r'(\bdata-iph="([^"]+)"[^<>]*?\bplaceholder=")[^"]*(")', placeholder, t)
    return "".join(parts)

def ld_extra(lang):
    qa = [{"@type": "Question", "name": text(f"faq_q{i}", lang), "acceptedAnswer": {"@type": "Answer", "text": text(f"faq_a{i}", lang)}} for i in range(1, 6)]
    faq = {"@type": "FAQPage", "inLanguage": lang, "mainEntity": qa}
    paper = {"@type": "ScholarlyArticle", "headline": "Reading the Architecture off the Weights: Manifest-Only Model Anatomy and a Census of 12,974 Hugging Face Repositories",
             "author": {"@id": "https://joessattes.github.io/Sattaya-AI-Website/#person"}, "datePublished": "2026-11", "inLanguage": "en",
             "url": "https://www.researchgate.net/publication/417293353", "sameAs": ["https://joessattes.github.io/ai-model-explorer/paper/manifest-anatomy-singkul-2026.pdf", "https://github.com/JoesSattes/ai-model-explorer/tree/main/paper"], "publisher": {"@type": "Organization", "name": "ResearchGate"}, "about": {"@id": BASE + "#app"}}
    return " " + json.dumps(faq, ensure_ascii=False) + ",\n " + json.dumps(paper, ensure_ascii=False)

for lang, p in PAGES.items():
    page = src
    for k, v in (("{{LANG}}", lang), ("{{ALT}}", p["alt"]), ("{{CANON}}", p["canon"]), ("{{ROOT}}", p["root"]),
                 ("{{OGLOC}}", p["ogloc"]), ("{{OGLOCALT}}", p["ogalt"]), ("{{LDEXTRA}}", ld_extra(lang)),
                 ("{{TITLE}}", html.escape(p["title"], quote=True)), ("{{DESC}}", html.escape(p["desc"], quote=True))):
        page = page.replace(k, v)
    head, body = page.split("<body>", 1)
    page = head + "<body>" + fill(body, lang)
    os.makedirs(os.path.dirname(os.path.abspath(p["out"])), exist_ok=True)
    open(p["out"], "w", encoding="utf-8").write(page)
    print(lang, p["out"], len(page))
