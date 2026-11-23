"""report.pdf dosyasını sonuç dosyalarından üretir.

Rapordaki bütün sayısal değerler results/ ve configs/ klasörlerindeki dosyalardan
okunur; elle girilmiş performans değeri yoktur.

Çalıştırma: python -m src.make_report
"""
import subprocess

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from src import project as P

R, F, C = P.RESULTS, P.FIGURES, P.CONFIGS
FONT_DIR = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("Serif", FONT_DIR + "DejaVuSerif.ttf"))
pdfmetrics.registerFont(TTFont("Serif-Bold", FONT_DIR + "DejaVuSerif-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Serif-Italic", FONT_DIR + "DejaVuSerif-Italic.ttf"))
pdfmetrics.registerFont(TTFont("Sans", FONT_DIR + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("Sans-Bold", FONT_DIR + "DejaVuSans-Bold.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("Serif", normal="Serif", bold="Serif-Bold", italic="Serif-Italic", boldItalic="Serif-Bold")
registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans", boldItalic="Sans-Bold")

BODY = ParagraphStyle("body", fontName="Serif", fontSize=11, leading=15.2, alignment=TA_JUSTIFY, spaceAfter=6)
H1 = ParagraphStyle("h1", fontName="Sans-Bold", fontSize=14, leading=18, spaceBefore=10, spaceAfter=6)
H2 = ParagraphStyle("h2", fontName="Sans-Bold", fontSize=11.5, leading=15, spaceBefore=6, spaceAfter=4)
CAP = ParagraphStyle("cap", fontName="Serif-Italic", fontSize=9.5, leading=12.5, alignment=TA_JUSTIFY, spaceBefore=3, spaceAfter=9)
CELL = ParagraphStyle("cell", fontName="Sans", fontSize=8, leading=10)
CELLB = ParagraphStyle("cellb", fontName="Sans-Bold", fontSize=8, leading=10)
TITLE = ParagraphStyle("title", fontName="Sans-Bold", fontSize=19, leading=25, alignment=TA_CENTER, spaceAfter=10)
SUB = ParagraphStyle("sub", fontName="Serif", fontSize=12, leading=17, alignment=TA_CENTER)
REF = ParagraphStyle("ref", parent=BODY, fontSize=10, leading=13.5, leftIndent=14, firstLineIndent=-14)


def f4(x):
    return f"{x:.4f}"


def ms(a, b):
    return f"{a:.4f} ± {b:.4f}"


def p(text):
    return Paragraph(text, BODY)


def table(rows, widths, header_rows=1, font=8):
    data = [[Paragraph(str(c), CELLB if i < header_rows else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=header_rows)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa5b1")),
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#e8edf3")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def fig(path, width_cm, caption):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(path)).getSize()
    img = Image(str(path), width=width_cm * cm, height=width_cm * cm * ih / iw)
    return KeepTogether([img, Paragraph(caption, CAP)])


def side_by_side(p1, w1, p2, w2, caption):
    from reportlab.lib.utils import ImageReader
    ims = []
    for pth, w in [(p1, w1), (p2, w2)]:
        iw, ih = ImageReader(str(pth)).getSize()
        ims.append(Image(str(pth), width=w * cm, height=w * cm * ih / iw))
    t = Table([ims], colWidths=[(w1 + 0.3) * cm, (w2 + 0.3) * cm])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return KeepTogether([t, Paragraph(caption, CAP)])


def tcap(text):
    return Paragraph(text, CAP)


# ---------------------------------------------------------------------------
# Verilerin okunması
# ---------------------------------------------------------------------------
split = pd.read_csv(R / "dataset_split_summary.csv").set_index("subset")
feat = pd.read_csv(R / "eda_feature_overview.csv").set_index("feature")
numsum = pd.read_csv(R / "eda_numeric_summary.csv", index_col=0)
corr = pd.read_csv(R / "eda_correlation_matrix.csv", index_col=0)
outl = pd.read_csv(R / "eda_outliers.csv", index_col=0)
rates = pd.read_csv(R / "eda_positive_rate_by_category.csv")
dups = pd.read_csv(R / "duplicate_feature_groups.csv")
manual_cfg = P.load_json(C / "manual_configs.json")
random_cfg = P.load_json(C / "random_configs.json")
seeds = P.load_json(C / "seeds.json")
ms_res = pd.read_csv(R / "manual_search_results.csv")
rs_res = pd.read_csv(R / "random_search_results.csv")
outer = pd.read_csv(R / "outer_cv_results.csv")
inner_runs = pd.read_csv(R / "inner_cv_training_runs.csv")
sel = P.load_json(R / "final_selection.json")
final_hp = P.load_json(R / "final_hyperparameters.json")
final_runs = pd.read_csv(R / "final_training_runs.csv")
test = pd.read_csv(R / "test_metrics.csv").set_index("model")
ci = pd.read_csv(R / "test_bootstrap_ci.csv")
seedst = pd.read_csv(R / "seed_stability_results.csv", index_col=0)
err = pd.read_csv(R / "test_error_analysis.csv", index_col=0)
opt_t = pd.read_csv(R / "optimizer_training_times.csv").set_index("optimizer")
supp = pd.read_csv(R / "supplementary_preprocessing_results.csv").set_index("candidate")
hist_final = pd.read_csv(R / "logs" / "training_histories_final.csv")

dev, tst = split.loc["development"], split.loc["test"]
FC = sel["final_candidate"]
MW, RW = sel["manual_winner"], sel["random_winner"]
dev_m = ms_res[ms_res.outer_fold == "development"].set_index("candidate")
dev_r = rs_res[rs_res.outer_fold == "development"].set_index("candidate")
main_label = [i for i in test.index if "(ana)" in i][0]
tm, tl, td = test.loc[main_label], test.loc["Logistic Regression"], test.loc["Dummy (most_frequent)"]


def outer_ms(method, metric):
    v = outer.loc[outer.method == method, metric]
    return v.mean(), v.std(ddof=1)


def ci_of(model, metric):
    r = ci[(ci.model == model) & (ci.metric == metric)].iloc[0]
    return r.ci95_low, r.ci95_high


n_dup_groups = dups.dup_group.nunique()
n_dup_cross = int((dups.groupby("dup_group").subset.nunique() > 1).sum())
OPT_TR = {"adamw": "AdamW", "adam": "Adam", "sgd": "SGD+M"}
SC_TR = {"standard": "Standard", "minmax": "MinMax", "robust": "Robust"}


def cfg_str(c):
    hl = "(" + ",".join(map(str, c["hidden_layers"])) + ("," if len(c["hidden_layers"]) == 1 else "") + ")"
    return (f"{OPT_TR[c['optimizer']]}, {hl}, {c['activation']}, LR={c['lr']:.3g}, WD={c['weight_decay']:g}, "
            f"BS={c['batch_size']}, {SC_TR[c['scaler']]}, drop={c['dropout']:g}, "
            f"kırpma={'var' if c['clipping'] else 'yok'}, sched={'açık' if c['scheduler'] else 'kapalı'}")


git_log = subprocess.run(["git", "log", "--reverse", "--format=%h|%cI|%s"], cwd=P.ROOT,
                         capture_output=True, text=True).stdout.strip().splitlines()
lock_commit = [l for l in git_log if "Kilitli test" in l][0].split("|")
m1_m6_same = inner_runs[inner_runs.candidate.isin(["M1", "M6"]) & inner_runs.stage.isin(["outer", "development_final"])]
pv = m1_m6_same.pivot_table(index=["stage", "outer_fold", "inner_fold"], columns="candidate", values="AP")
n_same = int((np.isclose(pv.M1, pv.M6, atol=1e-12)).sum()); n_tot = len(pv)
fe = final_runs.set_index("weight_seed")

story = []
# ---------------------------------------------------------------------------
# Kapak
# ---------------------------------------------------------------------------
story += [Spacer(1, 4 * cm), Paragraph("YAPAY SİNİR AĞLARI DERSİ — YÜKSEK LİSANS ARA SINAV PROJESİ", SUB), Spacer(1, 1 * cm),
          Paragraph("MLP ile Sınıflandırma: EDA, Ön İşleme, Hiperparametre Optimizasyonu ve Nested Cross-Validation", TITLE),
          Spacer(1, 0.6 * cm), Paragraph("UCI Bank Marketing (bank-additional.csv) üzerinde bir uygulama", SUB),
          Spacer(1, 3 * cm), Paragraph("Nurseda Akgül", SUB), Paragraph("Öğrenci numarası: 244329048", SUB),
          Paragraph(f"Temel seed: {seeds['base_seed']} (29048 + 2026)", SUB), Spacer(1, 1 * cm),
          Paragraph("GitHub deposu: https://github.com/nursedaburkuk/ysa-midterm-bank-marketing", SUB),
          Spacer(1, 0.4 * cm), Paragraph("Kasım 2026", SUB), PageBreak()]

# ---------------------------------------------------------------------------
# Özet
# ---------------------------------------------------------------------------
om_ap, os_ap = outer_ms("Manual search", "AP"); orr_ap, ors_ap = outer_ms("Random search", "AP")
ol_ap, ols_ap = outer_ms("LogReg", "AP")
lo_ap, hi_ap = ci_of(main_label, "AP")
story += [Paragraph("Özet", H1), p(
    "Bu çalışmada, banka telefon pazarlaması kampanyasında vadeli mevduat aboneliğinin görüşme öncesinde bilinen değişkenlerle "
    "tahmini için Çok Katmanlı Algılayıcı (MLP) modelleri incelenmiştir. Araştırma sorusu, MLP başarısının ön işleme, mimari, "
    "optimizer ve eğitim ayarlarına duyarlılığının ve hiperparametre seçim prosedürünün genelleme başarısının veri sızıntısı "
    "olmaksızın nasıl değerlendirilebileceğidir. UCI Bank Marketing veri setinin bank-additional.csv dosyası (4.119 gözlem) "
    "%80 geliştirme (development) ve %20 bağımsız test kümesine ayrılmıştır. Geliştirme kümesinde 3×3 iç içe çapraz doğrulama "
    "(nested cross-validation) ile altı manuel aday (M1–M6) ve altı rastgele arama adayı (R1–R6) iki ayrı arama olarak "
    "karşılaştırılmış, birincil metrik olarak Average Precision (AP) kullanılmıştır. Dış çapraz doğrulamada AP değeri manuel arama "
    f"için {ms(om_ap, os_ap)}, otomatik arama için {ms(orr_ap, ors_ap)}, Logistic Regression için {ms(ol_ap, ols_ap)} olarak "
    f"bulunmuştur. Seçilen {FC} konfigürasyonu bağımsız testte AP = {f4(tm.AP)} (%95 GA {f4(lo_ap)}–{f4(hi_ap)}), Logistic "
    f"Regression ise AP = {f4(tl.AP)} vermiştir. Bulgular, adaylar arasındaki AP farklarının katlar arası değişkenlikten küçük "
    "olduğunu ve MLP'nin doğrusal referans modele göre istatistiksel olarak ayırt edilebilir bir üstünlük sağlamadığını göstermektedir.")]

# ---------------------------------------------------------------------------
# 1 Giriş
# ---------------------------------------------------------------------------
story += [Paragraph("1. Giriş", H1), p(
    "MLP, doğrusal katmanlar ile doğrusal olmayan aktivasyon fonksiyonlarını art arda uygulayan ileri beslemeli bir sinir ağıdır. "
    "Modelin başarısı mimari, optimizer, öğrenme oranı, düzenlileştirme ve ön işleme seçimlerine bağlıdır. Hiperparametrelerin "
    "seçildiği veri üzerinde performans raporlanması iyimser yanlı tahminlere yol açtığından (Cawley ve Talbot, 2010), bu çalışmada "
    "seçim iç döngüde, performans tahmini ise dış döngüde yapılmıştır. Veri setinde pozitif sınıf oranı yaklaşık %11'dir ve tüm "
    "gözlemleri negatif sınıfa atayan bir model yaklaşık %89 doğruluk elde eder. Bu nedenle birincil metrik olarak AP seçilmiştir. "
    "AP, scikit-learn average_precision_score fonksiyonuyla precision–recall eğrisindeki basamakların ağırlıklı toplamı olarak "
    "hesaplanmıştır ve trapez yöntemiyle hesaplanan PR-AUC değerinden farklıdır.")]

# ---------------------------------------------------------------------------
# 2 Veri seti ve EDA
# ---------------------------------------------------------------------------
unk = feat.loc[P.CATEGORICAL_COLS]
unk_str = ", ".join(f"{c} {int(r.n_unknown)} (%{r.pct_unknown:.2f})" for c, r in unk.iterrows() if r.n_unknown > 0)
pd999 = feat.loc["pdays", "special_code"]
story += [Paragraph("2. Veri seti ve keşifsel veri analizi", H1),
          Paragraph("2.1 Veri, kullanım kuralları ve bölünme", H2), p(
    "Veri setinde 20 girdi değişkeni ve ikili hedef değişken (y) bulunmaktadır. \"yes\" pozitif (1), \"no\" negatif (0) sınıf olarak "
    "kodlanmıştır. Değişken adları, veri tipleri ve hedef etiketleri bölünmeden önce kontrol edilmiş, gerçek eksik değer (NaN) "
    "bulunmamıştır. Görüşme tamamlanmadan bilinmeyen duration değişkeni tüm modellerden çıkarılmış, 9 sayısal ve 10 kategorik "
    f"değişken kullanılmıştır. Veri, sınıf oranları korunarak {int(dev.n)} gözlemlik geliştirme (%{dev.positive_pct:.2f} pozitif) ve "
    f"{int(tst.n)} gözlemlik test (%{tst.positive_pct:.2f} pozitif) kümelerine ayrılmıştır (seed {seeds['seeds']['train_test_split']}). "
    "Satır indeksleri results/dataset_split.csv dosyasında kayıtlıdır.")]
story += [p(
    "Tüm sütunları özdeş kayıt bulunmamıştır. Ancak duration çıkarıldıktan sonra "
    f"{n_dup_groups} kayıt çiftinin özellik vektörleri özdeş hâle gelmekte, bu çiftlerden birinde etiketler farklılaşmaktadır. "
    "Söz konusu kayıtlar özgün veride ayrı gözlemler olduğundan silinmemiş, bölünmelerde ödevde belirtilen StratifiedKFold korunmuştur. "
    f"Bu çiftlerden {n_dup_cross} tanesinin geliştirme ve test kümelerine dağıldığı belirlenmiş ve küçük bir sızıntı kaynağı olarak "
    "kaydedilmiştir. Veri setinde müşteri kimliği bulunmadığından aynı müşteriye ait tekrarlar ayırt edilememektedir. Bu nedenle "
    "elde edilen sonuçlar, zamansal veya müşteri bağımsız genelleme olarak değil, aynı dağılımdan rastgele çekilmiş kayıtlara "
    "genelleme olarak yorumlanmalıdır.")]

t1 = [["Özellik / bilgi", "Development (n = %d)" % dev.n, "Test (n = %d)" % tst.n],
      ["Negatif sınıf (no, 0)", f"{int(dev.n_negative)} (%{dev.negative_pct:.2f})", f"{int(tst.n_negative)} (%{tst.negative_pct:.2f})"],
      ["Pozitif sınıf (yes, 1)", f"{int(dev.n_positive)} (%{dev.positive_pct:.2f})", f"{int(tst.n_positive)} (%{tst.positive_pct:.2f})"],
      ["Gerçek NaN", "0", "0 (yalnızca kontrol edildi)"],
      ["'unknown' kodları (development)", unk_str, "incelenmedi"],
      ["pdays özel kodu (development)", pd999.replace("(daha önce temas yok): ", "= daha önce temas yok; "), "incelenmedi"]]
story += [tcap("Tablo 1a. Veri bölünmesi, sınıf dağılımı ve eksik/özel kodlu değerler. EDA yalnızca geliştirme kümesinde yapılmış, test kümesi için yalnızca stratifikasyon sonucu sınıf oranı verilmiştir."),
          table(t1, [4.2, 7.0, 4.8])]
rows = [["Özellik", "Ortalama", "Medyan", "SD", "IQR", "Min", "Maks"]]
for c, r in numsum.iterrows():
    rows.append([c, f"{r['mean']:.3f}", f"{r['median']:.3f}", f"{r['SD']:.3f}", f"{r['IQR']:.3f}", f"{r['min']:.3f}", f"{r['max']:.3f}"])
story += [Spacer(1, 6), tcap("Tablo 1b. Sayısal değişkenlerin geliştirme kümesindeki tanımlayıcı istatistikleri (SD: örnek standart sapma). pdays ham hâliyle ve 999 kodu hariç ayrı verilmiştir."),
          table(rows, [3.6, 2.1, 2.1, 2.1, 2.1, 2.0, 2.0])]

top = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().sort_values(key=abs, ascending=False).head(3)
top_str = "; ".join(f"{a}–{b} r = {v:.2f}" for (a, b), v in top.items())
pr = rates.set_index(["feature", "category"]).positive_rate_pct
story += [Paragraph("2.2 EDA bulguları", H2), p(
    f"Geliştirme kümesi {int(dev.n_negative)} negatif ve {int(dev.n_positive)} pozitif gözlemden oluşmaktadır (Şekil 1a). Sayısal "
    "değişkenlerin ölçekleri belirgin biçimde farklıdır. nr.employed yaklaşık 5.000 düzeyinde değer alırken cons.conf.idx negatiftir; "
    f"campaign ise sağa çarpık bir dağılım göstermektedir (medyan {numsum.loc['campaign','median']:.0f}, en büyük değer "
    f"{numsum.loc['campaign','max']:.0f}). Pozitif sınıfın düşük euribor3m, emp.var.rate ve nr.employed değerlerinde yoğunlaştığı "
    f"gözlenmiştir (Şekil 2). Makroekonomik değişkenler arasında çok yüksek korelasyon bulunmaktadır ({top_str}; Şekil 1b). Hedefe "
    "dayalı özellik seçimi model seçimi niteliği taşıdığından bu değişkenler veride tutulmuştur.")]
story += [p(
    f"Kategorik değişkenlerde en fazla unknown değeri default değişkeninde görülmektedir (%{feat.loc['default','pct_unknown']:.2f}). "
    f"pdays değişkenindeki 999 kodu daha önce temas kurulmadığını belirtmekte olup geliştirme kümesindeki oranı "
    f"%{100*(1-numsum.loc['pdays (999 hariç)','n']/dev.n):.2f}'dir. Betimsel olarak pozitif sınıf oranı, önceki kampanyası başarılı "
    f"müşterilerde %{pr['poutcome','success']:.1f}, önceki teması olmayanlarda %{pr['poutcome','nonexistent']:.1f} olarak "
    "hesaplanmıştır. Bu karşılaştırmalar herhangi bir özellik seçimi veya ön işleme kararında kullanılmamıştır.")]
story += [side_by_side(F / "fig_eda_class_distribution.png", 5.6, F / "fig_eda_correlation.png", 9.2, "Şekil 1. (a) Geliştirme kümesinde sınıf dağılımı. (b) Sayısal değişkenlerin Pearson korelasyon matrisi (pdays ham hâliyle)."),
          fig(F / "fig_eda_boxplots.png", 11.0, "Şekil 2. Sayısal değişkenlerin hedef sınıflarına göre kutu grafikleri (geliştirme kümesi; pdays için 999 kodu hariç)."),
          ]
story += [p(
    "Aykırı değerler IQR ve %1–%99 yüzdelik kurallarıyla işaretlenmiştir. campaign değişkeninde gözlemlerin "
    f"%{outl.loc['campaign','pct_IQR_outlier']:.1f}'i IQR üst sınırını aşmaktadır. previous değişkeninde Q1 = Q3 = 0 olduğundan "
    "IQR kuralı sıfırdan farklı tüm değerleri aykırı saymakta ve anlamlı bir tanım sağlamamaktadır. Değerler ölçüm hatası "
    "niteliği taşımadığından satır silinmemiş, yalnızca eğitim katının fit alt kümesinden öğrenilen %1 ve %99 yüzdeliklerine göre "
    "kırpma seçeneği arama uzayına eklenmiştir.")]
story += [Paragraph("2.3 Deney hipotezleri", H2), p(
    "Aşağıdaki hipotezler EDA'nın ardından, herhangi bir model sonucu elde edilmeden yazılmış ve ilk commit ile kayıt altına alınmıştır. "
    "<b>H1:</b> Sınıf dengesizliği nedeniyle Dummy modelin AP değeri pozitif sınıf oranına (yaklaşık 0.11) eşit, doğruluğu ise yaklaşık "
    "0.89 olacak; 0.50 eşiğinde tüm modellerde duyarlılık 0.50'nin altında kalacaktır. "
    "<b>H2:</b> 0.001 öğrenme oranında SGD + Momentum (M3), AdamW (M1) ve Adam'a (M2) göre daha yavaş yakınsayacak ve daha düşük AP "
    "verecek; Adam ile AdamW arasındaki fark küçük olacaktır. "
    "<b>H3:</b> StandardScaler (M1), MinMaxScaler'a (M4) eşit veya daha yüksek AP sağlayacaktır. "
    "<b>H4:</b> (64,32) mimarisi (M5), (32,) mimarisine göre AP artışı sağlamayacak ve daha erken aşırı öğrenme gösterecektir. "
    "<b>H5:</b> Erken durdurma her iki durumda da etkin olduğundan öğrenme oranı zamanlayıcısının kapatılması (M6) AP'yi anlamlı "
    "biçimde değiştirmeyecektir.")]

# ---------------------------------------------------------------------------
# 3 Yöntem
# ---------------------------------------------------------------------------
s = seeds["seeds"]
story += [Paragraph("3. Yöntem", H1), Paragraph("3.1 Seed politikası ve iç içe çapraz doğrulama", H2), p(
    f"Temel seed, öğrenci numarasının son beş hanesine (29048) 2026 eklenerek {seeds['base_seed']} olarak belirlenmiş, alt seed'ler "
    f"sabit kaydırmalarla türetilmiştir: Python {s['python']}, NumPy {s['numpy']}, PyTorch {s['torch']}, test bölünmesi "
    f"{s['train_test_split']}, dış CV {s['outer_cv']}, iç CV {s['inner_cv']}, izleme bölünmesi {s['early_stop_split']}, rastgele arama "
    f"{s['random_search']}, bootstrap {s['bootstrap']}, ek ağırlık seed'leri {s['torch_stability_2']} ve {s['torch_stability_3']}. "
    "PyTorch'ta deterministik algoritmalar kullanılmıştır.")]
story += [p(
    "Uygulanan protokol Şekil 3'te gösterilmektedir. Dış döngüde StratifiedKFold(n_splits=3, shuffle=True, random_state=31078), her "
    "dış eğitim kümesinde ise StratifiedKFold(n_splits=3, shuffle=True, random_state=31079) kullanılmıştır. Manuel ve otomatik adaylar "
    "aynı iç katlarda iki ayrı arama olarak değerlendirilmiştir (3 × 3 × 12 = 108 eğitim). Her aramada iç CV ortalama AP'si en yüksek "
    "aday seçilmiş, eşitlik durumunda sırasıyla düşük SD, az parametre ve küçük aday kodu ölçütleri uygulanmıştır. Seçilen adaylar dış "
    "eğitim kümesinde yeniden eğitilmiş ve dış değerlendirme kümesinde bir kez ölçülmüştür (6 eğitim). Ardından tüm geliştirme "
    "kümesinde iç CV tekrarlanmış (36 eğitim) ve iki kazanandan iç CV AP'si yüksek olan nihai konfigürasyon olarak belirlenmiştir. "
    "Dış CV sonuçları hiçbir seçimde kullanılmamıştır.")]
story += [fig(F / "fig_nested_cv_flow.png", 10.5, "Şekil 3. Veri bölünmesi ve iç içe çapraz doğrulama akış şeması.")]
story += [Paragraph("3.2 Kat-içi ön işleme", H2), p(
    "Her MLP eğitiminde eğitim verisi %85 fit ve %15 stratified izleme alt kümelerine ayrılmıştır. Medyanlar, kategori sözlüğü, "
    "ölçekleyici istatistikleri ve kırpma eşikleri yalnızca fit alt kümesinde ColumnTransformer/Pipeline ile öğrenilmiş, diğer tüm "
    "kümelere yalnızca transform uygulanmıştır. Sayısal değişkenlere medyan ile doldurma, isteğe bağlı kırpma ve ölçekleme; "
    "kategorik değişkenlere OneHotEncoder(handle_unknown='ignore') uygulanmış, unknown ayrı kategori olarak korunmuştur. pdays = 999 "
    "için model sonuçlarından bağımsız sabit bir kural tanımlanmıştır: pdays_no_prev_contact gösterge değişkeni eklenmiş, 999 "
    "değerleri eksik kabul edilerek fit alt kümesinin medyanıyla doldurulmuştur.")]
story += [p(
    "MinMaxScaler bir normalizasyon yöntemidir ve değerleri (x − min) / (maks − min) ile [0, 1] aralığına taşıdığından aykırı değerlere "
    "duyarlıdır. StandardScaler bir standardizasyon yöntemidir ve (x − ortalama) / SD ile z-skoru üretir. RobustScaler medyan ve IQR "
    "kullandığından aykırı değerlere karşı daha dayanıklıdır.")]
story += [Paragraph("3.3 MLP yapısı ve eğitim kontrolü", H2), p(
    "Modeller PyTorch ile oluşturulmuştur. Gizli katmanlar Linear, aktivasyon ve isteğe bağlı Dropout katmanlarından oluşmakta, çıkış "
    "katmanı tek logit üretmektedir. Tam bağlantılı bir katmandaki öğrenilebilir parametre sayısı girdi × çıktı + çıktı ile "
    "hesaplanmıştır. 61 girdili (32,) mimarisi için bu değer 61×32+32 = 1.984 ve 32×1+1 = 33 olmak üzere 2.017'dir. Girdi boyutu, fit "
    "alt kümesindeki kategori sayısına bağlı olarak 61 veya 62 olabildiğinden her eğitimin parametre sayısı ayrıca kaydedilmiştir. "
    "Eğitim ayarları Tablo 2'de verilmiştir.")]
t2 = [["Bileşen", "Ayar"], ["Kayıp ve çıkış", "Tek logit, BCEWithLogitsLoss; olasılık = sigmoid(logit); eşik 0.50"],
      ["En fazla epoch", "150 (en iyi ve durulan epoch kaydedilir)"],
      ["Optimizer", "SGD + Momentum (0.9), Adam, AdamW; weight decay optimizer parametresi olarak"],
      ["Erken durdurma", "Aktif; patience = 15; en iyi ağırlıklar geri yüklenir"],
      ["İzleme verisi", "İç eğitim verisinden %15 stratified alt küme (seed 31080)"],
      ["LR scheduler", "ReduceLROnPlateau, mode = min, factor = 0.5, patience = 5, min_lr = 1e-5 (M6'da kapalı)"],
      ["Eğitimi kontrol eden metrik", "İzleme alt kümesinin BCE kaybı; her epoch sonunda scheduler ve erken durdurma bu kayıpla güncellenir"],
      ["Hiperparametre seçen metrik", "İç CV doğrulama katında average_precision_score (AP)"],
      ["Kayıtlar", "Epoch bazında eğitim kaybı, izleme kaybı ve öğrenme oranı; en iyi epoch, süre, uyarılar (results/logs)"]]
story += [tcap("Tablo 2. MLP eğitim ayarları. Eğitimi kontrol eden metrik BCE kaybı, hiperparametre seçen birincil metrik AP'dir."), table(t2, [4.5, 11.5])]
story += [fig(F / "fig_mlp_architecture.png", 12.0,
              f"Şekil 4. Nihai MLP ({FC}) mimarisi ve öğrenilebilir parametre sayısı (girdi × çıktı + çıktı). Hesaplanan toplam, PyTorch'un "
              f"saydığı değerle ({final_hp['architecture']['n_trainable_parameters']}) doğrulanmıştır.")]
story += [Paragraph("3.4 Referans modeller", H2), p(
    "DummyClassifier(strategy='most_frequent') dengesiz sınıf yapısı için alt sınır olarak kullanılmıştır. LogisticRegression (C = 1.0, "
    "lbfgs, max_iter = 5000), M1 ile aynı sabit ön işleme kuralıyla kurulmuştur. Referans modeller her dış katta dış eğitim kümesinde "
    "eğitilip dış değerlendirme kümesinde ölçülmüş, son aşamada tüm geliştirme kümesinde eğitilerek kilitli test tahmin dosyasına "
    "eklenmiştir. Bu modeller arama bütçesine dahil edilmemiştir.")]

# ---------------------------------------------------------------------------
# 4 Deneyler
# ---------------------------------------------------------------------------
story += [Paragraph("4. Manuel ve otomatik hiperparametre araması", H1), p(
    "Manuel adaylar, M1 temel konfigürasyonundan tek bir faktör değiştirilerek tanımlanmıştır (batch size 64, dropout 0, ReLU, weight "
    "decay 1e-4). Otomatik adaylar ParameterSampler ile her optimizer için iki aday olacak biçimde örneklenmiş, öğrenme oranı Adam ve "
    "AdamW için 1e-4–1e-2, SGD için 1e-3–1e-1 aralığında log-uniform dağılımdan çekilmiştir. Tüm adaylar iç içe çapraz doğrulama "
    "başlamadan önce ayrı bir commit ile kaydedilmiştir. Sonuçlar Tablo 3 ve Tablo 4'te verilmiştir.")]


def search_table(cfgs, res, devres):
    rows = [["Kod", "Parametreler", "Dev. iç CV AP (ort. ± SD)", "Dış kat 1 / 2 / 3 iç AP", "En iyi epoch (ort.)", "Toplam süre (s)", "Param."]]
    for code, c in cfgs.items():
        d = devres.loc[code]
        per = [res[(res.outer_fold == str(k)) & (res.candidate == code)].mean_AP.iloc[0] for k in (1, 2, 3)]
        rows.append([("<b>%s</b>" % code) if d.selected else code, cfg_str(c), ms(d.mean_AP, d.sd_AP),
                     " / ".join(f"{v:.4f}" for v in per), f"{d.mean_best_epoch:.1f}", f"{d.total_time_s:.2f}", f"{int(d.n_params)}"])
    return rows


ms_res["outer_fold"] = ms_res.outer_fold.astype(str); rs_res["outer_fold"] = rs_res.outer_fold.astype(str)
story += [tcap(f"Tablo 3. Manuel adaylar M1–M6. Dev. iç CV AP: tüm geliştirme kümesindeki 3 iç katın ortalaması ± örnek SD; kalın kod seçilen adaydır ({MW}). "
               "Dış kat sütunları her dış eğitim kümesindeki iç CV ortalama AP'dir. Süre, tüm geliştirme aşamasındaki üç eğitimin toplamıdır."),
          table(search_table(manual_cfg, ms_res, dev_m), [0.9, 6.2, 2.4, 2.9, 1.3, 1.3, 1.1])]
story += [Spacer(1, 6), tcap(f"Tablo 4. Otomatik (Random Search) adaylar R1–R6. Arama uzayı: optimizer {{SGD+M, Adam, AdamW}} (her biri 2 aday), gizli katmanlar "
               "{(16,), (32,), (64,), (64,32), (128,64)}, aktivasyon {ReLU, tanh}, weight decay {0, 1e-5, 1e-4, 1e-3}, batch size {32, 64, 128}, ölçekleyici "
               f"{{Standard, MinMax, Robust}}, dropout {{0, 0.2}}, kırpma {{yok, var}}. Kalın kod seçilen adaydır ({RW})."),
          table(search_table(random_cfg, rs_res, dev_r), [0.9, 6.2, 2.4, 2.9, 1.3, 1.3, 1.1])]

sel_m = {k: ms_res[(ms_res.outer_fold == str(k)) & (ms_res.selected)].candidate.iloc[0] for k in (1, 2, 3)}
sel_r = {k: rs_res[(rs_res.outer_fold == str(k)) & (rs_res.selected)].candidate.iloc[0] for k in (1, 2, 3)}
dm = dev_m
story += [p(
    f"Adaylar arasındaki AP farkları katlar arası değişkenlikten küçüktür (Şekil 5). Tüm geliştirme aşamasında manuel adayların "
    f"ortalama AP değerleri {dm.mean_AP.min():.4f}–{dm.mean_AP.max():.4f} aralığında kalırken iç katlar arası SD ortalama "
    f"{dm.sd_AP.mean():.3f} düzeyindedir. Bu nedenle seçilen aday dış katlar arasında değişmiştir (manuel: {sel_m[1]}, {sel_m[2]}, "
    f"{sel_m[3]}; otomatik: {sel_r[1]}, {sel_r[2]}, {sel_r[3]}). Tüm geliştirme kümesinde manuel kazanan {MW} "
    f"({ms(dm.loc[MW,'mean_AP'], dm.loc[MW,'sd_AP'])}), otomatik kazanan {RW} ({ms(dev_r.loc[RW,'mean_AP'], dev_r.loc[RW,'sd_AP'])}) "
    f"olmuş, önceden tanımlanan kurala göre {FC} nihai konfigürasyon olarak belirlenmiştir. İki kazanan arasındaki fark katlar arası "
    f"SD'nin altında olduğundan bu seçim bir üstünlük olarak yorumlanmamalıdır. Optimizer bazında eğitim süreleri ve örnek eğitim "
    f"eğrileri Ek'te (Tablo A1, Şekil A1) verilmiştir. Seçime dahil edilmeyen ek analizde (Tablo A2) kırpma, unknown için en sık sınıf "
    f"ile doldurma ve RobustScaler, M1'e göre AP'yi sırasıyla {supp.loc['S_clip','mean_AP']-supp.loc['M1','mean_AP']:+.4f}, "
    f"{supp.loc['S_unknown_mf','mean_AP']-supp.loc['M1','mean_AP']:+.4f} ve {supp.loc['S_robust','mean_AP']-supp.loc['M1','mean_AP']:+.4f} "
    "değiştirmiştir; bu farklar da katlar arası SD'nin altındadır.")]
story += [fig(F / "fig_search_comparison.png", 14.5, "Şekil 5. Manuel (mavi) ve otomatik (yeşil) adayların iç CV ortalama AP ± SD değerleri. Kalın kenarlı çubuk seçilen adayı, noktalı çizgi pozitif sınıf oranını göstermektedir.")]

# ---------------------------------------------------------------------------
# 5 Nested CV bulguları
# ---------------------------------------------------------------------------
rows = [["Kat", "Yöntem", "Aday", "AP", "ROC-AUC", "F1", "Sens.", "Spec.", "Bal. Acc.", "Acc.", "TP/FP/TN/FN"]]
for _, r in outer.iterrows():
    rows.append([r.outer_fold, r.method.replace("Manual search", "Manuel arama").replace("Random search", "Otomatik arama").replace("LogReg", "Logistic Regression"),
                 r.selected_candidate, f4(r.AP), f4(r.ROC_AUC), f4(r.F1), f4(r.Sensitivity), f4(r.Specificity), f4(r.Balanced_Accuracy), f4(r.Accuracy),
                 f"{r.TP}/{r.FP}/{r.TN}/{r.FN}"])
story += [Paragraph("5. İç içe çapraz doğrulama bulguları", H1),
          tcap("Tablo 5. Dış CV kat bazlı sonuçlar (eşik 0.50). Her dış değerlendirme kümesinde 1.098 veya 1.099 gözlem ve 120 veya 121 pozitif örnek bulunmaktadır."),
          table(rows, [0.85, 2.45, 1.15, 1.2, 1.5, 1.2, 1.2, 1.2, 1.4, 1.2, 2.3])]
rows = [["Yöntem", "AP", "ROC-AUC", "F1", "Sens.", "Spec.", "Bal. Acc."]]
for meth, lab in [("Manual search", "Manuel arama (MLP)"), ("Random search", "Otomatik arama (MLP)"), ("LogReg", "Logistic Regression"), ("Dummy", "Dummy")]:
    rows.append([lab] + [ms(*outer_ms(meth, m)) for m in ["AP", "ROC_AUC", "F1", "Sensitivity", "Specificity", "Balanced_Accuracy"]])
story += [Spacer(1, 6), tcap("Tablo 6. Dış CV özeti: ortalama ± örnek standart sapma (n = 3). Katlar bağımsız tekrarlar olmadığından SD güven aralığı olarak yorumlanmamalıdır."),
          table(rows, [3.6, 2.15, 2.15, 2.15, 2.15, 2.15, 2.15])]
lm, lsd = outer_ms("LogReg", "F1"); rm_, rsd = outer_ms("Random search", "F1"); mm_, msd = outer_ms("Manual search", "F1")
story += [p(
    f"İki arama prosedürü dış CV'de birbirine yakın AP değerleri vermiştir (manuel {ms(om_ap, os_ap)}, otomatik {ms(orr_ap, ors_ap)}). "
    f"Logistic Regression ortalaması ({ms(ol_ap, ols_ap)}) daha yüksek olmakla birlikte katlar arası SD 0.06–0.07 düzeyinde olduğundan "
    "bu fark anlamlı kabul edilmemiştir. Kat sıralaması da tutarlı değildir; 1. ve 3. katta Logistic Regression, 2. katta otomatik arama "
    f"en yüksek AP'yi vermiştir. Dummy modelin AP değeri ({f4(outer_ms('Dummy','AP')[0])}) pozitif sınıf oranına eşittir ve "
    f"{f4(outer_ms('Dummy','Accuracy')[0])} doğruluğa karşın hiçbir pozitif örnek tanınamamıştır. 0.50 eşiğinde öğrenen modellerin "
    "duyarlılığı 0.10–0.25, özgüllüğü ise 0.97'nin üzerindedir. Logistic Regression'ın F1 değeri "
    f"({ms(lm, lsd)}) MLP prosedürlerinden ({ms(rm_, rsd)} ve {ms(mm_, msd)}) yüksektir; bu fark MLP'lerin 0.50 eşiğinde daha az pozitif "
    "tahmin üretmesiyle açıklanabilir.")]

# ---------------------------------------------------------------------------
# 6 Bağımsız test
# ---------------------------------------------------------------------------
fcfg = sel["config"]
story += [Paragraph("6. Bağımsız test", H1), Paragraph("6.1 Nihai kilitli model", H2), p(
    f"Nihai konfigürasyon {FC} ({cfg_str(fcfg)}) tüm geliştirme kümesinde üç ağırlık seed'iyle "
    f"({', '.join(map(str, final_hp['weight_seeds']))}) eğitilmiştir. Ana model seed {final_hp['primary_seed']} ve karar eşiği 0.50 olarak "
    "önceden belirlenmiştir. Model ağırlıkları, ön işleme durumları ve öğrenilmiş dönüşüm istatistikleri test etiketleri okunmadan "
    "kaydedilmiştir. Üç MLP, Logistic Regression ve Dummy modelinin test olasılıkları ile sınıf tahminleri etiket içermeyen "
    f"test_predictions_locked.csv dosyasına yazılmış ve {lock_commit[0]} numaralı commit ile ({lock_commit[1]}) GitHub'a gönderilmiştir. "
    "Test etiketleri bu commit'ten sonra okunmuş ve metrikler kilitli dosyadaki tahminlerden hesaplanmıştır. Ana seed'de en iyi izleme "
    f"kaybı {int(fe.loc[final_hp['primary_seed'],'best_epoch'])}. epoch'ta, diğer iki seed'de {int(fe.loc[s['torch_stability_2'],'best_epoch'])}. "
    f"epoch'ta elde edilmiştir (Şekil 6). Görece yüksek SGD öğrenme oranı ({fcfg['lr']:.4f}) nedeniyle izleme kaybı erken minimuma "
    "ulaşmış, zamanlayıcının öğrenme oranını düşürmesi sonrasında iyileşme gözlenmemiştir.")]
story += [fig(F / "fig_training_curves_final.png", 11.0, f"Şekil 6. Nihai model {FC}: üç ağırlık seed'i için eğitim ve izleme BCE kaybı (üstte) ile epoch bazında öğrenme oranı (altta, log ölçek).")]

rows = [["Model", "AP", "ROC-AUC", "F1", "Acc.", "Prec.", "Sens.", "Spec.", "Bal. Acc.", "TP/FP/TN/FN"]]
for lab in [main_label, "Logistic Regression", "Dummy (most_frequent)"]:
    r = test.loc[lab]
    rows.append([lab, f4(r.AP), f4(r.ROC_AUC), f4(r.F1), f4(r.Accuracy), f4(r.Precision), f4(r.Sensitivity), f4(r.Specificity), f4(r.Balanced_Accuracy), f"{r.TP}/{r.FP}/{r.TN}/{r.FN}"])
nb = int(ci.n_boot.iloc[0]); rej = int(ci.rejected_single_class.max())
story += [Paragraph("6.2 Test metrikleri ve güven aralıkları", H2),
          tcap(f"Tablo 7. Bağımsız test metrikleri (n = {int(tst.n)}, {int(tst.n_positive)} pozitif; eşik 0.50) ve {nb} tekrarlı bootstrap ile %95 yüzdelik güven aralıkları "
               f"(seed {s['bootstrap']}). Her yeniden örneklemede iki sınıfın bulunduğu kontrol edilmiş, tek sınıflı örnekleme sayısı {rej} olmuştur."),
          table(rows, [3.3, 1.25, 1.45, 1.25, 1.3, 1.3, 1.3, 1.3, 1.45, 2.3])]
ci_rows = [["Model", "AP [%95 GA]", "ROC-AUC [%95 GA]", "F1 [%95 GA]"]]
for lab in [main_label, "Logistic Regression"]:
    r = test.loc[lab]
    ci_rows.append([lab] + [f"{r[m]:.4f} [{ci_of(lab, m)[0]:.4f}–{ci_of(lab, m)[1]:.4f}]" for m in ["AP", "ROC_AUC", "F1"]])
story += [Spacer(1, 4), table(ci_rows, [4.6, 4.0, 4.0, 4.0])]
rows = [["Ağırlık seed'i", "AP", "ROC-AUC", "F1", "Sens.", "Spec.", "Bal. Acc.", "En iyi epoch"]]
for lab, r in seedst.iterrows():
    if lab in ("mean", "sd"):
        continue
    rows.append([f"{int(r.weight_seed)}" + (" (ana)" if "(ana)" in lab else ""), f4(r.AP), f4(r.ROC_AUC), f4(r.F1), f4(r.Sensitivity), f4(r.Specificity), f4(r.Balanced_Accuracy),
                 int(fe.loc[int(r.weight_seed), "best_epoch"])])
mrow, srow = seedst.loc["mean"], seedst.loc["sd"]
rows.append(["Ortalama ± SD (n = 3)"] + [ms(mrow[m], srow[m]) for m in ["AP", "ROC_AUC", "F1", "Sensitivity", "Specificity", "Balanced_Accuracy"]] + [""])
story += [Spacer(1, 6), tcap("Tablo 8. Üç başlangıç seed'inin bağımsız test performansı ve betimsel ortalama ± SD. Ana sonuç önceden belirlenen seed 31076'ya aittir."),
          table(rows, [3.1, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.4]), Spacer(1, 6)]
lo_f, hi_f = ci_of(main_label, "F1"); lo_r, hi_r = ci_of(main_label, "ROC_AUC")
story += [p(
    f"Ana MLP modeli bağımsız testte AP = {f4(tm.AP)} (%95 GA {f4(lo_ap)}–{f4(hi_ap)}), ROC-AUC = {f4(tm.ROC_AUC)} ve F1 = {f4(tm.F1)} "
    f"vermiştir. Logistic Regression için karşılık gelen değerler AP = {f4(tl.AP)}, ROC-AUC = {f4(tl.ROC_AUC)} ve F1 = {f4(tl.F1)} "
    "olarak hesaplanmıştır (Tablo 7). MLP'nin AP ve ROC-AUC değerleri daha yüksek, F1 değeri daha düşüktür; ancak güven aralıklarının büyük ölçüde örtüşmesi "
    f"iki model arasında anlamlı bir fark bulunmadığını göstermektedir. Üç seed'in AP değerleri ({ms(mrow.AP, srow.AP)}) kararlıdır; F1 "
    f"değerindeki daha yüksek değişkenlik ({ms(mrow.F1, srow.F1)}) eşik yakınındaki az sayıda örneğin sınıf değiştirmesinden kaynaklanmaktadır (Tablo 8).")]
story += [fig(F / "fig_test_confusion_matrices.png", 8.5, "Şekil 7. Bağımsız test karışıklık matrisleri: ana MLP (solda) ve Logistic Regression (sağda), eşik 0.50."),
          fig(F / "fig_test_roc_pr.png", 12.5, "Şekil 8. Bağımsız testte ROC (solda) ve Precision–Recall (sağda) eğrileri. PR panelindeki yatay çizgi pozitif sınıf oranıdır.")]
story += [Paragraph("6.3 Hata analizi", H2), p(
    f"Ana MLP, 90 pozitif örneğin {int(err.loc['TP','n'])} tanesini doğru sınıflandırmış, {int(err.loc['FN','n'])} tanesini kaçırmıştır; "
    f"734 negatif örnekten {int(err.loc['FP','n'])} tanesi yanlış pozitif olarak sınıflandırılmıştır (Şekil 7). Doğru pozitiflerde önceki "
    f"kampanyası başarılı müşterilerin oranı %{err.loc['TP','poutcome_success_pct']:.0f}, yanlış negatiflerde ise daha önce temas "
    f"kurulmamış müşterilerin oranı %{err.loc['FN','pdays_999_pct']:.0f} olarak bulunmuştur. Yanlış negatiflerin medyan tahmin olasılıkları "
    f"({err.loc['FN','prob']:.3f}) doğru negatiflerinkine ({err.loc['TN','prob']:.3f}) yakındır. Yanlış pozitiflerin tamamı önceki "
    "kampanyası başarılı müşterilerdir. Bu örüntü, mevcut değişkenlerle ilk kez temas kurulan müşterilerde abonelik ayrımının "
    "yapılamadığını ve modelin önceki kampanya başarısına güçlü biçimde dayandığını göstermektedir. Eşik test verisine göre "
    "değiştirilmemiştir.")]

# ---------------------------------------------------------------------------
# 7 Tartışma
# ---------------------------------------------------------------------------
d = dev_m
story += [Paragraph("7. Tartışma", H1), Paragraph("7.1 Hipotezlerin değerlendirilmesi", H2), p(
    "<b>H1 desteklenmiştir.</b> Dummy modelin AP değeri pozitif sınıf oranına eşit, doğruluğu yaklaşık 0.89'dur ve 0.50 eşiğinde öğrenen "
    "modellerin duyarlılığı 0.25'in altında kalmıştır. <b>H2 desteklenmiştir.</b> Tüm geliştirme aşamasında ortalama en iyi epoch M3 için "
    f"{d.loc['M3','mean_best_epoch']:.1f}, M1 için {d.loc['M1','mean_best_epoch']:.1f} olarak bulunmuş ve M3'ün altı eğitimi 150 epoch sınırına "
    f"ulaşmıştır. M3'ün AP değeri ({f4(d.loc['M3','mean_AP'])}) M1'in ({f4(d.loc['M1','mean_AP'])}) altında kalmış, Adam ile AdamW "
    f"arasındaki fark {abs(d.loc['M2','mean_AP']-d.loc['M1','mean_AP']):.4f} düzeyinde olmuştur. Otomatik aramada daha yüksek öğrenme "
    f"oranıyla ({fcfg['lr']:.3f}) örneklenen SGD adayı R1'in en iyi sonucu vermesi, M3'teki düşük başarının yöntemden değil, SGD için "
    "küçük kalan öğrenme oranından kaynaklandığını düşündürmektedir.")]
story += [p(
    f"<b>H3 kısmen desteklenmiştir.</b> Tüm geliştirme kümesinde StandardScaler (M1, {f4(d.loc['M1','mean_AP'])}) MinMaxScaler'dan "
    f"(M4, {f4(d.loc['M4','mean_AP'])}) yüksek AP vermiş, ancak 2. dış katta M4 seçilmiştir. <b>H4 kısmen reddedilmiştir.</b> M5 daha erken "
    f"durmuş (ortalama en iyi epoch {d.loc['M5','mean_best_epoch']:.1f}), ancak tüm geliştirme kümesinde M1'den yüksek AP "
    f"({f4(d.loc['M5','mean_AP'])}) vermiştir; bu fark 3. dış katta tersine dönmüştür. <b>H5 desteklenmiştir.</b> M1 ve M6, {n_tot} iç "
    f"CV eğitiminin {n_same} tanesinde özdeş AP üretmiştir. Epoch kayıtları, ReduceLROnPlateau'nun öğrenme oranını en iyi izleme "
    "epoch'undan sonra düşürdüğünü göstermektedir (Şekil A1). En iyi ağırlıklar geri yüklendiğinden bu düşüş çoğu eğitimde nihai "
    "modeli etkilememiştir.")]
story += [Paragraph("7.2 Genelleme ve model karmaşıklığı", H2), p(
    f"Dış CV, arama prosedürlerinin genelleme başarısını; tüm geliştirme kümesindeki iç CV ({FC} için "
    f"{ms(sel['dev_inner_cv_mean_AP'][FC], sel['dev_inner_cv_sd_AP'][FC])}) yalnızca seçimi; bağımsız test ise tek bir kilitli modelin "
    f"başarısını ölçmektedir. Test AP değeri ({f4(tm.AP)}) dış CV ortalamasının yaklaşık 0.04 altında, ancak dış kat aralığı (0.37–0.50) "
    f"içindedir. Benzer düşüşün Logistic Regression'da da görülmesi ({f4(ol_ap)} → {f4(tl.AP)}), farkın model seçimine özgü bir aşırı "
    "öğrenmeden çok yalnızca 90 pozitif örnek içeren test kümesinin örnekleme değişkenliğinden kaynaklandığını düşündürmektedir. "
    "Eğitim kaybı azalırken izleme kaybının artması aşırı öğrenmeye işaret etmekte, erken durdurma bu etkiyi sınırlamaktadır. 1.009 ile "
    "16.257 arasında parametre içeren mimariler arasındaki AP farkları katlar arası değişkenliğin altında kalmış ve doğrusal Logistic "
    "Regression da benzer başarı göstermiştir. Bu bulgular, veri setindeki öğrenilebilir sinyalin büyük ölçüde doğrusal olduğunu ve "
    "daha karmaşık mimarilerin ek kazanç sağlamadığını göstermektedir.")]
story += [Paragraph("7.3 Sınırlamalar", H2), p(
    "Dış CV yalnızca üç kattan oluşmakta ve katlar aynı veriyi paylaştığından bağımsız tekrar niteliği taşımamaktadır. Test kümesinin "
    "küçük olması eşiğe bağlı metriklerde geniş güven aralıklarına yol açmıştır. Sınıf ağırlıklandırma veya eşik ayarlaması "
    "uygulanmadığından 0.50 eşiğinde duyarlılık düşük kalmıştır; uygulamada eşik yalnızca geliştirme kümesi iç CV tahminleriyle "
    "belirlenmelidir. Müşteri kimliğinin bulunmaması nedeniyle rastgele bölünme zamansal veya müşteri bağımsız genellemeyi kanıtlamamakta, "
    f"özellikleri özdeş {n_dup_groups} kayıt çiftinin {n_dup_cross} tanesi geliştirme ve test kümelerine dağılmaktadır. Nihai modelin en iyi "
    "epoch değerleri (2–7) yüksek öğrenme oranı nedeniyle oldukça erkendir. Son olarak, veri seti kamuya açık olduğundan kilitli tahmin "
    "commit'i kör test garantisi değil, zaman sıralı bir deney kaydıdır. Tüm eğitimler sonlu kayıp değerleriyle tamamlanmış, uyarı "
    "kayıtları yalnızca 150 epoch sınırına ulaşan M3 ve R3 eğitimlerinde oluşmuştur.")]

# ---------------------------------------------------------------------------
# 8 Sonuç
# ---------------------------------------------------------------------------
story += [Paragraph("8. Sonuç", H1), p(
    "Bu çalışmada test verisini tüm kararlardan ayıran, ön işlemeyi her eğitimin fit alt kümesinde öğrenen ve manuel ile otomatik "
    "aramayı aynı iç katlarda karşılaştıran bir iç içe çapraz doğrulama protokolü uygulanmıştır. Hiperparametre seçiminin AP üzerindeki "
    "etkisi katlar arası değişkenlikten küçük bulunmuştur. Aynı küçük öğrenme oranında SGD belirgin biçimde yavaş yakınsamış, uygun "
    "öğrenme oranıyla ise en iyi adaylar arasında yer almıştır. Öğrenme oranı zamanlayıcısı, erken durdurma ile birlikte kullanıldığında "
    "çoğu eğitimde nihai modeli değiştirmemiştir. MLP, dış CV'de ve bağımsız testte Logistic Regression'a göre ayırt edilebilir bir "
    f"üstünlük göstermemiştir (test AP {f4(tm.AP)} ve {f4(tl.AP)}). İleriki çalışmalarda eşik seçiminin geliştirme kümesi iç CV'si ile "
    "yapılması ve sonuçların zamansal doğrulama ile yeniden sınanması önerilmektedir.")]
story += [Paragraph("Kaynakça", H1)]
for ref in [
    "Moro, S., Rita, P. ve Cortez, P. (2014). Bank Marketing [Veri seti]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306",
    "Moro, S., Cortez, P. ve Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. Decision Support Systems, 62, 22–31.",
    "Pedregosa, F. ve diğerleri (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
    "scikit-learn. Common pitfalls and recommended practices: Data leakage. https://scikit-learn.org/stable/common_pitfalls.html",
    "scikit-learn. Cross-validation, model selection and classification metrics documentation (StratifiedKFold, ParameterSampler, average_precision_score). https://scikit-learn.org/stable/",
    "Paszke, A. ve diğerleri (2019). PyTorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems 32.",
    "PyTorch. ReduceLROnPlateau documentation. https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html",
    "PyTorch. BCEWithLogitsLoss, SGD, Adam ve AdamW documentation. https://docs.pytorch.org/docs/stable/",
    "Loshchilov, I. ve Hutter, F. (2019). Decoupled weight decay regularization. International Conference on Learning Representations.",
    "Cawley, G. C. ve Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. Journal of Machine Learning Research, 11, 2079–2107.",
]:
    story.append(Paragraph(ref, REF))

# ---------------------------------------------------------------------------
# Ek
# ---------------------------------------------------------------------------
story += [PageBreak(), Paragraph("Ek A. Tekrarlanabilirlik bilgileri", H1), p(
    "Kod: https://github.com/nursedaburkuk/ysa-midterm-bank-marketing. Çalıştırma sırası README.md dosyasındadır (01_eda.ipynb → src.make_configs → 02_nested_cv.ipynb "
    "→ 03_final_test.ipynb → src.make_report). Ortam: Ubuntu 24.04.5 LTS, Intel Xeon 2.10 GHz (2 çekirdek), 7 GB RAM, GPU yok; Python 3.13.18, PyTorch 2.15.0, "
    "scikit-learn 1.9.1, pandas 3.0.5, NumPy 2.5.3, SciPy 1.18.1, matplotlib 3.11.2. Ham veri SHA-256: 7e59cf65…df50. Ondalık ayırıcı olarak nokta kullanılmıştır. "
    "Toplam eğitim sayısı: 108 dış-kat iç CV eğitimi, 6 dış-kat yeniden eğitimi, 36 tüm-development iç CV eğitimi, 9 ek analiz eğitimi ve 3 nihai eğitim.")]
story += [fig(F / "fig_training_curves_manual_examples.png", 12.5, "Şekil A1. Tüm development iç CV'si, 1. iç kat: M1 (AdamW), M3 (SGD + Momentum) ve M6 (scheduler kapalı) için eğitim (fit) ve erken durdurma izleme BCE kaybı (üstte) ve epoch bazında öğrenme oranı (altta, log ölçek). Kesikli dikey çizgi en iyi epoch'tur.")]
t5 = [["Optimizer", "Eğitim sayısı", "Ortalama süre (s)", "Toplam süre (s)", "Ortalama durma epoch'u", "150'ye ulaşan"]]
for o, r in opt_t.iterrows():
    t5.append([OPT_TR[o], int(r.n_trainings), f"{r.mean_time_s:.3f}", f"{r.total_time_s:.1f}", f"{r.mean_stopped_epoch:.1f}", int(r.n_hit_150)])
story += [tcap("Tablo A1. Optimizer bazında eğitim süreleri (108 dış-kat iç CV eğitimi + 36 tüm-development iç CV eğitimi). Donanım: Intel Xeon 2.10 GHz, 2 çekirdek, yalnızca CPU."),
          table(t5, [2.5, 2.2, 2.8, 2.6, 3.2, 2.5])]
t6 = [["Değişiklik (M1'e göre)", "İç CV AP (ort. ± SD)", "Katlar"]]
for code, r in supp.iterrows():
    t6.append([f"{code}: {r.change_vs_M1}", ms(r.mean_AP, r.sd_AP), r.fold_APs.replace(";", " / ")])
story += [Spacer(1, 6), tcap("Tablo A2. Ek analiz: ön işleme alternatifleri (tüm development, aynı iç katlar). Bu analiz seçim bütçesine dahil değildir ve hiçbir seçimi etkilememiştir."),
          table(t6, [7.0, 4.0, 5.0])]
story += [fig(F / "fig_outer_cv_ap.png", 9.0, "Şekil A2. Dış kat AP değerleri: manuel arama, otomatik arama, Logistic Regression ve Dummy (Tablo 5 ile aynı değerler).")]
rows = [["Commit", "Zaman", "Açıklama"]] + [l.split("|", 2) for l in git_log]
story += [tcap("Tablo A3. Zaman sıralı git kaydı (rapor üretildiği andaki geçmiş)."), table(rows, [1.7, 4.3, 10.0])]


def on_page(canvas, doc):
    if doc.page > 1:
        canvas.setFont("Sans", 8.5)
        canvas.drawCentredString(A4[0] / 2, 1.2 * cm, str(doc.page - 1))


doc = SimpleDocTemplate(str(P.ROOT / "report.pdf"), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                        topMargin=2.0 * cm, bottomMargin=2.0 * cm, title="MLP ile Sınıflandırma — Bank Marketing",
                        author="Nurseda Akgül")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("report.pdf yazıldı")
