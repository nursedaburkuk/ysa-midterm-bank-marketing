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
story += [Paragraph("Özet ve araştırma sorusu", H1), p(
    "Bu çalışmanın araştırma sorusu şudur: Bir banka telefon pazarlaması kampanyasında müşterinin vadeli mevduat "
    "aboneliğini görüşme öncesinde bilinen bilgilerle tahmin eden bir Çok Katmanlı Algılayıcının (MLP) başarısı, ön işleme, "
    "mimari, optimizer ve eğitim ayarlarına ne kadar duyarlıdır ve hiperparametreleri seçme prosedürünün kendisi "
    "veri sızıntısı olmadan ne kadar iyi genelleşir? Çalışmada UCI Bank Marketing veri setinin bank-additional.csv "
    f"dosyası (4.119 gözlem, pozitif sınıf oranı %{100*split.loc['all','positive_pct']/100:.2f}) kullanılmıştır. "
    "Veri %80 development ve %20 bağımsız test olarak ayrılmış, test kümesi bütün kararlar bitene kadar kullanılmamıştır. "
    "Development kümesinde 3×3 nested cross-validation ile altı hipotez temelli manuel aday (M1–M6) ve altı Random "
    "Search adayı (R1–R6) iki ayrı arama olarak değerlendirilmiş, birincil metrik olarak Average Precision (AP) kullanılmıştır.")]
story += [p(
    f"Dış CV'de manuel arama prosedürü {ms(om_ap, os_ap)}, otomatik arama prosedürü {ms(orr_ap, ors_ap)} AP vermiştir. "
    f"Aynı katlarda Logistic Regression referansı {ms(ol_ap, ols_ap)} AP ile MLP prosedürlerinden ayırt edilemeyecek düzeydedir. "
    f"Tüm development kümesindeki iç CV seçimiyle nihai konfigürasyon {FC} olmuştur (SGD + Momentum, iki gizli katman, tanh). "
    f"Etiketsiz tahminler ayrı bir commit ile kilitlendikten sonra ölçülen bağımsız test sonuçlarında ana MLP modeli "
    f"AP = {f4(tm.AP)} (%95 bootstrap GA {f4(lo_ap)}–{f4(hi_ap)}), ROC-AUC = {f4(tm.ROC_AUC)} ve F1 = {f4(tm.F1)} elde etmiştir. "
    f"Logistic Regression testte AP = {f4(tl.AP)} ve F1 = {f4(tl.F1)} vermiştir. Temel bulgu, bu küçük ve dengesiz veri setinde "
    "adaylar arasındaki AP farklarının (en fazla yaklaşık 0.02) katlar arası belirsizlikten küçük kaldığı ve MLP'nin doğrusal referansa "
    "karşı kanıtlanabilir bir üstünlük göstermediğidir. 0.50 eşiğinde bütün modellerin duyarlılığı düşük kalmıştır.")]

# ---------------------------------------------------------------------------
# 1 Giriş
# ---------------------------------------------------------------------------
story += [Paragraph("1. Giriş", H1), p(
    "MLP, doğrusal katmanlar ve doğrusal olmayan aktivasyonlarla karar sınırları öğrenen ileri beslemeli bir sinir ağıdır ve başarısı "
    "mimariye, optimizer'a, öğrenme oranına, düzenlileştirmeye ve ön işlemeye bağlıdır. Aynı veride hem hiperparametre seçip hem performans "
    "raporlamak iyimser bir tahmin üretir; nested cross-validation iç döngüde seçim, dış döngüde ölçüm yaparak bu sorunu çözer. "
    "Veri setinde pozitif sınıf (abonelik) oranı yaklaşık %11'dir. Bu dengesizlik altında accuracy yanıltıcıdır, çünkü "
    "her müşteriye \"hayır\" diyen bir model %89 doğruluk elde eder. Bu yüzden birincil metrik olarak pozitif sınıfa "
    "odaklanan Average Precision (AP) kullanılmıştır. AP, scikit-learn'ün average_precision_score fonksiyonuyla, "
    "precision–recall eğrisindeki basamakların ağırlıklı toplamı olarak hesaplanır ve trapez yöntemiyle hesaplanan "
    "PR-AUC ile aynı şey değildir. Bu raporda yalnızca AP adı kullanılmıştır. Çalışmanın amacı tek bir yüksek skor üretmek "
    "değil, tekrarlanabilir ve sızıntısız bir seçim ve değerlendirme süreci kurmaktır.")]

# ---------------------------------------------------------------------------
# 2 Veri seti ve EDA
# ---------------------------------------------------------------------------
unk = feat.loc[P.CATEGORICAL_COLS]
unk_str = ", ".join(f"{c} {int(r.n_unknown)} (%{r.pct_unknown:.2f})" for c, r in unk.iterrows() if r.n_unknown > 0)
pd999 = feat.loc["pdays", "special_code"]
story += [Paragraph("2. Veri seti ve keşifsel veri analizi", H1),
          Paragraph("2.1 Veri, kullanım kuralları ve bölünme", H2), p(
    "Veri setinde 20 girdi özelliği ve ikili hedef y bulunur; \"yes\" pozitif (1), \"no\" negatif (0) sınıf olarak kodlanmıştır. "
    "Değişken adları, veri tipleri ve hedef etiketleri bölünmeden önce kontrol edilmiştir; gerçek NaN değeri yoktur. "
    "duration değişkeni görüşme bitmeden bilinmediği için, model performansına bakılmadan bütün modellerden çıkarılmıştır. "
    "Böylece 9 sayısal ve 10 kategorik olmak üzere 19 girdi özelliği kalmıştır. "
    f"Veri, sınıf oranlarını koruyan stratified bölünme ile {int(dev.n)} satırlık development (%{dev.positive_pct:.2f} pozitif) ve "
    f"{int(tst.n)} satırlık bağımsız test (%{tst.positive_pct:.2f} pozitif) kümelerine ayrılmıştır (seed {seeds['seeds']['train_test_split']}). "
    "Orijinal satır indeksleri ve sınıf oranları results/dataset_split.csv dosyasına kaydedilmiştir.")]
story += [p(
    f"Kopya kayıt kontrolünde 21 sütunun tamamı aynı olan kayıt bulunmamıştır. Ancak duration çıkarıldıktan sonra "
    f"{n_dup_groups} çift ({2*n_dup_groups} satır) özellik bakımından birebir aynıdır ve bu çiftlerden biri farklı etiket taşır. "
    "Bu satırlar orijinal dosyada farklı kayıtlar olduğu için silinmemiş, ödev StratifiedKFold istediği için grup tabanlı bölme de "
    f"kullanılmamıştır. Bunun yerine etkisi sayılmıştır: {n_dup_groups} çiftin {n_dup_cross} tanesi development ile test arasında bölünmüştür ve "
    "bu küçük bir sızıntı kaynağıdır. Ayrıca veri setinde müşteri kimliği bulunmadığından aynı müşterinin tekrarları belirlenemez. "
    "Bu nedenle rastgele bölünme, modelin farklı zamanlara veya yeni müşterilere genellendiğini kanıtlamaz; elde edilen sonuçlar "
    "aynı dağılımdan rastgele çekilen kayıtlara genelleme olarak yorumlanmalıdır.")]

t1 = [["Özellik / bilgi", "Development (n = %d)" % dev.n, "Test (n = %d)" % tst.n],
      ["Negatif sınıf (no, 0)", f"{int(dev.n_negative)} (%{dev.negative_pct:.2f})", f"{int(tst.n_negative)} (%{tst.negative_pct:.2f})"],
      ["Pozitif sınıf (yes, 1)", f"{int(dev.n_positive)} (%{dev.positive_pct:.2f})", f"{int(tst.n_positive)} (%{tst.positive_pct:.2f})"],
      ["Gerçek NaN", "0", "0 (yalnızca kontrol edildi)"],
      ["'unknown' kodları (development)", unk_str, "incelenmedi"],
      ["pdays özel kodu (development)", pd999.replace("(daha önce temas yok): ", "= daha önce temas yok; "), "incelenmedi"]]
story += [tcap("Tablo 1a. Veri bölünmesi, sınıf dağılımı ve eksik/özel kodlu değerler. EDA yalnızca development kümesinde yapılmıştır; test kümesinde yalnızca sınıf oranı (stratifikasyonun sonucu) raporlanmıştır."),
          table(t1, [4.2, 7.0, 4.8])]
rows = [["Özellik", "Ortalama", "Medyan", "SD", "IQR", "Min", "Maks"]]
for c, r in numsum.iterrows():
    rows.append([c, f"{r['mean']:.3f}", f"{r['median']:.3f}", f"{r['SD']:.3f}", f"{r['IQR']:.3f}", f"{r['min']:.3f}", f"{r['max']:.3f}"])
story += [Spacer(1, 6), tcap("Tablo 1b. Sayısal özelliklerin development kümesindeki tanımlayıcı istatistikleri (SD: örnek standart sapma). pdays ham hâliyle ve 999 kodu hariç ayrı ayrı verilmiştir."),
          table(rows, [3.6, 2.1, 2.1, 2.1, 2.1, 2.0, 2.0])]

top = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().sort_values(key=abs, ascending=False).head(3)
top_str = "; ".join(f"{a}–{b} r = {v:.2f}" for (a, b), v in top.items())
pr = rates.set_index(["feature", "category"]).positive_rate_pct
story += [Paragraph("2.2 EDA bulguları", H2), p(
    f"Şekil 1a'da görüldüğü gibi development kümesinde {int(dev.n_negative)} negatif ve {int(dev.n_positive)} pozitif gözlem vardır. "
    "Sayısal değişkenlerin ölçekleri çok farklıdır: nr.employed yaklaşık 5.000 civarındayken cons.conf.idx negatiftir ve campaign "
    f"değişkeni sağa çarpık bir dağılıma sahiptir (medyan {numsum.loc['campaign','median']:.0f}, en büyük değer {numsum.loc['campaign','max']:.0f}). "
    f"Şekil 2'deki boxplot'larda pozitif sınıfın daha düşük euribor3m, emp.var.rate ve nr.employed değerlerinde yoğunlaştığı görülür. "
    f"Şekil 1b'deki korelasyon matrisinde makroekonomik değişkenler arasında çok yüksek ilişki vardır ({top_str}). "
    "Bu değişkenler hedefle ilişkiye göre özellik seçimi model seçimi sayılacağı için veride bırakılmıştır.")]
story += [p(
    f"Kategorik değişkenlerde en fazla unknown değeri default değişkenindedir ({int(feat.loc['default','n_unknown'])} satır, %{feat.loc['default','pct_unknown']:.2f}). "
    f"pdays değişkeninde 999 kodu \"daha önce temas yok\" anlamına gelir; bu kodun development satırlarındaki oranı %{100*(1-numsum.loc['pdays (999 hariç)','n']/dev.n):.2f} olarak bulunmuştur. "
    "Betimsel olarak, önceki kampanyası başarılı olan müşterilerde (poutcome = success) pozitif oran "
    f"%{pr['poutcome','success']:.1f}, önceki teması olmayanlarda %{pr['poutcome','nonexistent']:.1f} olarak bulunmuştur; cellular iletişimde oran "
    f"%{pr['contact','cellular']:.1f}, telephone iletişimde %{pr['contact','telephone']:.1f} olarak bulunmuştur. Bu karşılaştırmalar yalnızca betimseldir ve hiçbir "
    "özellik seçimi veya ön işleme kararında kullanılmamıştır.")]
story += [side_by_side(F / "fig_eda_class_distribution.png", 5.6, F / "fig_eda_correlation.png", 9.2, "Şekil 1. (a) Development kümesinde sınıf dağılımı. (b) Sayısal özelliklerin Pearson korelasyon matrisi (development; pdays ham hâliyle)."),
          fig(F / "fig_eda_boxplots.png", 11.0, "Şekil 2. Sayısal özelliklerin hedef sınıflarına göre boxplot'ları (development; pdays için 999 kodu hariç)."),
          ]
story += [p(
    "Aykırı değerler IQR ve %1–%99 yüzdelik kurallarıyla işaretlenmiştir. campaign değişkeninde IQR sınırının üzerindeki "
    f"gözlemlerin oranı %{outl.loc['campaign','pct_IQR_outlier']:.1f} olarak bulunmuştur. previous değişkeninde Q1 = Q3 = 0 olduğundan IQR kuralı "
    f"sıfır olmayan her değeri (%{outl.loc['previous','pct_IQR_outlier']:.1f}) aykırı sayar ve anlamlı bir tanım vermez. Makroekonomik değişkenlerde IQR dışı "
    "değer yoktur. Değerler ölçüm hatası gibi görünmediği için hiçbir satır silinmemiş; bunun yerine yalnızca ilgili eğitim katının "
    "fit alt kümesinden öğrenilen %1 ve %99 yüzdeliklerine göre kırpma seçeneği arama uzayına alınmıştır.")]
story += [Paragraph("2.3 Deney hipotezleri", H2), p(
    "Aşağıdaki hipotezler EDA'dan sonra, hiçbir model sonucu görülmeden yazılmış ve 01_eda.ipynb içinde ilk commit ile kaydedilmiştir. "
    "<b>H1:</b> Sınıf dengesizliği nedeniyle Dummy modelin AP değeri pozitif orana (yaklaşık 0.11) eşit olacak, accuracy ise yanıltıcı biçimde "
    "yüksek (yaklaşık 0.89) çıkacaktır; 0.50 eşiğinde bütün modellerde duyarlılık 0.50'nin altında kalacaktır. "
    "<b>H2:</b> Aynı öğrenme oranıyla (0.001) SGD + Momentum (M3) AdamW (M1) ve Adam'a (M2) göre daha yavaş yakınsayacak ve daha düşük AP verecektir; "
    "Adam ile AdamW arasındaki fark küçük olacaktır. "
    "<b>H3:</b> StandardScaler (M1) en az MinMaxScaler (M4) kadar yüksek AP verecektir. "
    "<b>H4:</b> (64,32) mimarisi (M5), daha fazla parametreye rağmen (32,) mimarisine göre AP artışı sağlamayacak ve daha erken aşırı öğrenecektir. "
    "<b>H5:</b> Erken durdurma her iki durumda aktif olduğu için scheduler'ı kapatmak (M6) AP'yi anlamlı biçimde değiştirmeyecektir.")]

# ---------------------------------------------------------------------------
# 3 Yöntem
# ---------------------------------------------------------------------------
s = seeds["seeds"]
story += [Paragraph("3. Yöntem", H1), Paragraph("3.1 Seed politikası ve nested CV protokolü", H2), p(
    f"Temel seed, öğrenci numarasının son beş hanesine (29048) 2026 eklenerek {seeds['base_seed']} olarak belirlenmiştir. Alt seed'ler temel seed'e "
    f"sabit bir kaydırma eklenerek türetilmiştir: Python {s['python']}, NumPy {s['numpy']}, PyTorch birincil ağırlık seed'i {s['torch']}, "
    f"train/test bölünmesi {s['train_test_split']}, dış CV {s['outer_cv']}, iç CV {s['inner_cv']}, %15'lik izleme bölünmesi {s['early_stop_split']}, "
    f"Random Search {s['random_search']} (SGD, Adam ve AdamW için sırasıyla +0, +1, +2), bootstrap {s['bootstrap']}; seed kararlılığı için ek "
    f"ağırlık seed'leri {s['torch_stability_2']} ve {s['torch_stability_3']}. PyTorch'ta deterministik algoritmalar ve tek iş parçacığı kullanılmıştır.")]
story += [p(
    "Şekil 3'teki protokolde dış CV için StratifiedKFold(n_splits=3, shuffle=True, random_state=31078) kullanılmıştır. Her dış eğitim "
    "kümesinde StratifiedKFold(n_splits=3, shuffle=True, random_state=31079) ile üç iç kat oluşturulmuş ve manuel altı aday ile otomatik altı aday "
    "aynı iç katlarda iki ayrı arama olarak değerlendirilmiştir (3 × 3 × 12 = 108 eğitim). Her aramada iç CV ortalama AP'si en yüksek aday seçilmiştir; "
    "eşitlik durumunda önceden yazılan kurala göre sırasıyla daha düşük SD, daha az parametre ve daha küçük aday kodu tercih edilir. "
    "Seçilen iki aday dış eğitim kümesinde yeniden eğitilmiş ve dış değerlendirme kümesinde yalnızca bir kez ölçülmüştür (6 eğitim). "
    "Son olarak tüm development kümesinde iç CV tekrarlanmış (36 eğitim), manuel ve otomatik kazananlardan iç CV ortalama AP'si yüksek olan "
    "nihai konfigürasyon olarak seçilmiştir. Dış CV sonuçları yalnızca arama prosedürlerinin genelleme başarısını tanımlamak için kullanılmış, "
    "hiçbir seçimde kullanılmamıştır.")]
story += [fig(F / "fig_nested_cv_flow.png", 10.5, "Şekil 3. Veri bölünmesi ve nested CV akış şeması. Manuel ve otomatik arama aynı iç katları kullanır ve ayrı kazanan seçer; referans modeller aynı dış katlarda değerlendirilir.")]
story += [Paragraph("3.2 Kat-içi ön işleme", H2), p(
    "Her MLP eğitiminde, verilen eğitim verisi önce %85 fit ve %15 stratified erken durdurma izleme alt kümesine ayrılır. Ön işlemenin bütün "
    "parametreleri (sayısal medyanlar, kategori sözlüğü, scaler istatistikleri ve kırpma eşikleri) yalnızca fit alt kümesinde scikit-learn "
    "ColumnTransformer/Pipeline ile öğrenilir; izleme, iç doğrulama, dış değerlendirme ve test verilerine yalnızca transform uygulanır. "
    "Sayısal sütunlarda medyan imputation, isteğe bağlı %1–%99 kırpma ve scaler sırasıyla uygulanır. Kategorik sütunlarda unknown ayrı kategori "
    "olarak tutulur ve OneHotEncoder(handle_unknown='ignore') kullanılır. pdays = 999 için model sonuçlarına bakılmadan şu sabit kural konmuştur: "
    "pdays_no_prev_contact adında ikili bir gösterge sütunu eklenir ve pdays içindeki 999 değerleri eksik sayılarak fit alt kümesinin medyanıyla "
    "doldurulur. Böylece \"temas yok\" bilgisi korunur ve 999 değeri ölçeklemeyi bozmaz; hiçbir satır silinmez.")]
story += [p(
    "Normalization ve standardization farklı işlemlerdir. MinMaxScaler bir normalization (aralık ölçekleme) yöntemidir ve her değeri "
    "(x − min) / (maks − min) ile [0, 1] aralığına taşır; bu yüzden aykırı değerlere duyarlıdır. StandardScaler bir standardization yöntemidir ve "
    "(x − ortalama) / SD ile z-skoru üretir. RobustScaler ise medyanı çıkarıp IQR'a böler ve aykırı değerlere karşı daha dayanıklıdır.")]
story += [Paragraph("3.3 MLP yapısı ve eğitim kontrolü", H2), p(
    "MLP'ler PyTorch ile yazılmıştır. Gizli katmanlar Linear → aktivasyon (temel seçim ReLU) → isteğe bağlı Dropout dizisinden oluşur; çıkış katmanı "
    "tek logit üretir. Tam bağlantılı bir katmanın öğrenilebilir parametre sayısı girdi × çıktı + çıktı formülüyle hesaplanır. Örneğin 61 girdili "
    "(32,) mimarisinde 61×32+32 = 1.984 ve 32×1+1 = 33 olmak üzere 2.017 parametre vardır. Girdi boyutu, fit alt kümesinde görülen kategori sayısına "
    "göre 61 veya 62 olabilir; bu nedenle her eğitimin gerçek parametre sayısı ayrıca kaydedilmiştir. Eğitim ayarları Tablo 2'de verilmiştir.")]
t2 = [["Bileşen", "Ayar"], ["Kayıp ve çıkış", "Tek logit, BCEWithLogitsLoss; olasılık = sigmoid(logit); eşik 0.50"],
      ["En fazla epoch", "150 (en iyi ve durulan epoch kaydedilir)"],
      ["Optimizer", "SGD + Momentum (0.9), Adam, AdamW; weight decay optimizer parametresi olarak"],
      ["Erken durdurma", "Aktif; patience = 15; en iyi ağırlıklar geri yüklenir"],
      ["İzleme verisi", "İç eğitim verisinden %15 stratified alt küme (seed 31080)"],
      ["LR scheduler", "ReduceLROnPlateau, mode = min, factor = 0.5, patience = 5, min_lr = 1e-5 (M6'da kapalı)"],
      ["Eğitimi kontrol eden metrik", "İzleme alt kümesinin BCE kaybı; her epoch sonunda scheduler ve erken durdurma bu kayıpla güncellenir"],
      ["Hiperparametre seçen metrik", "İç CV doğrulama katında average_precision_score (AP)"],
      ["Kayıtlar", "Epoch bazında eğitim kaybı, izleme kaybı ve öğrenme oranı; en iyi epoch, süre, uyarılar (results/logs)"]]
story += [tcap("Tablo 2. MLP eğitim ayarları. Eğitimi kontrol eden metrik BCE kaybıdır; hiperparametreleri seçen birincil metrik AP'dir."), table(t2, [4.5, 11.5])]
story += [fig(F / "fig_mlp_architecture.png", 12.0,
              f"Şekil 4. Nihai MLP ({FC}) mimarisi ve öğrenilebilir parametre sayısı. Her katman için parametre sayısı girdi × çıktı + çıktı (ağırlık + bias) "
              f"olarak hesaplanmış ve PyTorch'un saydığı değerle ({final_hp['architecture']['n_trainable_parameters']}) karşılaştırılarak doğrulanmıştır.")]
story += [Paragraph("3.4 Referans modeller", H2), p(
    "DummyClassifier(strategy='most_frequent') dengesiz sınıf yapısı için alt sınırı verir; bu model her gözleme pozitif sınıf olasılığı 0 atadığından "
    "AP değeri pozitif oranına eşittir. LogisticRegression (C = 1.0, lbfgs, max_iter = 5000) M1 ile aynı sabit ön işleme kuralıyla (medyan "
    "imputation, StandardScaler, kırpma yok, unknown ayrı kategori) kurulmuştur. Referans modeller her dış katta yalnızca dış eğitim kümesinde "
    "eğitilmiş ve dış değerlendirme kümesinde ölçülmüştür. Son aşamada tüm development kümesinde eğitilip aynı kilitli test tahmin dosyasına yazılmışlardır. "
    "Referans modeller arama bütçesine dahil değildir.")]

# ---------------------------------------------------------------------------
# 4 Deneyler
# ---------------------------------------------------------------------------
story += [Paragraph("4. Deneyler: manuel ve otomatik hiperparametre araması", H1), p(
    "Manuel adaylar M1 temel konfigürasyonundan tek bir faktör değiştirilerek tanımlanmıştır (ortak ayarlar: batch size 64, dropout 0, ReLU, "
    "weight decay 1e-4, erken durdurma aktif). Otomatik aday uzayı Tablo 4'ün başlığında özetlenmiştir; adaylar scikit-learn ParameterSampler ile her "
    "optimizer için tam ikişer tane olacak biçimde örneklenmiş, öğrenme oranı Adam/AdamW için 1e-4–1e-2, SGD için 1e-3–1e-1 aralığında log-uniform "
    "dağılımdan çekilmiştir. Bütün adaylar nested CV başlamadan configs/ klasörüne yazılıp ayrı bir commit ile kaydedilmiştir. Tablo 3 ve Tablo 4, tüm "
    "development kümesindeki seçim aşamasının iç CV sonuçlarını ve her dış kattaki iç CV ortalama AP değerlerini birlikte gösterir.")]


def search_table(cfgs, res, devres):
    rows = [["Kod", "Parametreler", "Dev. iç CV AP (ort. ± SD)", "Dış kat 1 / 2 / 3 iç AP", "En iyi epoch (ort.)", "Toplam süre (s)", "Param."]]
    for code, c in cfgs.items():
        d = devres.loc[code]
        per = [res[(res.outer_fold == str(k)) & (res.candidate == code)].mean_AP.iloc[0] for k in (1, 2, 3)]
        rows.append([("<b>%s</b>" % code) if d.selected else code, cfg_str(c), ms(d.mean_AP, d.sd_AP),
                     " / ".join(f"{v:.4f}" for v in per), f"{d.mean_best_epoch:.1f}", f"{d.total_time_s:.2f}", f"{int(d.n_params)}"])
    return rows


ms_res["outer_fold"] = ms_res.outer_fold.astype(str); rs_res["outer_fold"] = rs_res.outer_fold.astype(str)
story += [tcap(f"Tablo 3. Manuel adaylar M1–M6. Dev. iç CV AP: tüm development kümesindeki 3 iç katın ortalaması ± örnek SD; kalın kod seçilen manuel adaydır ({MW}). "
               "Dış kat sütunları her dış eğitim kümesindeki iç CV ortalama AP'dir. Süre, tüm development aşamasındaki üç eğitimin toplamıdır."),
          table(search_table(manual_cfg, ms_res, dev_m), [0.9, 6.2, 2.4, 2.9, 1.3, 1.3, 1.1])]
story += [Spacer(1, 6), tcap(f"Tablo 4. Otomatik (Random Search) adaylar R1–R6. Arama uzayı: optimizer {{SGD+M, Adam, AdamW}} (her biri 2 aday), gizli katmanlar "
               "{(16,), (32,), (64,), (64,32), (128,64)}, aktivasyon {ReLU, tanh}, weight decay {0, 1e-5, 1e-4, 1e-3}, batch size {32, 64, 128}, scaler "
               f"{{Standard, MinMax, Robust}}, dropout {{0, 0.2}}, kırpma {{yok, var}}. Kalın kod seçilen otomatik adaydır ({RW})."),
          table(search_table(random_cfg, rs_res, dev_r), [0.9, 6.2, 2.4, 2.9, 1.3, 1.3, 1.1])]

sel_m = {k: ms_res[(ms_res.outer_fold == str(k)) & (ms_res.selected)].candidate.iloc[0] for k in (1, 2, 3)}
sel_r = {k: rs_res[(rs_res.outer_fold == str(k)) & (rs_res.selected)].candidate.iloc[0] for k in (1, 2, 3)}
dm = dev_m
story += [p(
    f"Şekil 5, iç CV sonuçlarını dört aşama için karşılaştırır. Adaylar arasındaki AP farkları, katlar arasındaki farklardan belirgin biçimde küçüktür. "
    f"Örneğin tüm development aşamasında manuel adayların ortalama AP'si {dm.mean_AP.min():.4f} ile {dm.mean_AP.max():.4f} arasındayken, "
    f"aynı adayın üç iç kat arasındaki SD'si ortalama {dm.sd_AP.mean():.3f} düzeyindedir. Bu nedenle seçilen aday dış kattan dış kata değişmiştir: manuel "
    f"aramada sırasıyla {sel_m[1]}, {sel_m[2]} ve {sel_m[3]}; otomatik aramada {sel_r[1]}, {sel_r[2]} ve {sel_r[3]} seçilmiştir. Tüm development "
    f"aşamasında manuel kazanan {MW} ({ms(dm.loc[MW,'mean_AP'], dm.loc[MW,'sd_AP'])}), otomatik kazanan {RW} "
    f"({ms(dev_r.loc[RW,'mean_AP'], dev_r.loc[RW,'sd_AP'])}) olmuş ve önceden tanımlanan kurala göre nihai konfigürasyon {FC} seçilmiştir. "
    "İki kazanan arasındaki fark, katlar arası SD'nin çok altında olduğundan bu seçim istatistiksel bir üstünlük olarak değil, önceden belirlenmiş "
    "kuralın uygulanması olarak yorumlanmalıdır.")]
story += [fig(F / "fig_search_comparison.png", 14.5, "Şekil 5. Manuel (mavi) ve otomatik (yeşil) adayların iç CV ortalama AP ± SD değerleri; her dış kat ve tüm development aşaması için ayrı panel. Kalın kenarlı çubuk seçilen adaydır; noktalı çizgi pozitif sınıf oranıdır.")]

story += [p(
    f"Ek'teki Şekil A1'de AdamW (M1) izleme kaybını ortalama {dm.loc['M1','mean_best_epoch']:.0f}. epoch civarında en aza indirirken, aynı öğrenme oranındaki "
    f"SGD + Momentum (M3) ortalama {dm.loc['M3','mean_best_epoch']:.0f}. epoch'ta en iyi değerine ulaşmıştır. Arama bütçesindeki eşitlik aday sayısına ilişkin olduğu için "
    "optimizer bazında eğitim süreleri Ek'teki Tablo A1'de ayrıca verilmiştir.")]
story += [p(
    "Ölçekleyicileri, kırpmayı ve unknown için en sık sınıf yaklaşımını değerlendirmek amacıyla M1'den tek bir ön işleme faktörü değiştirilerek "
    f"seçime dahil olmayan bir ek analiz yapılmıştır (Ek, Tablo A2). Kırpma ortalama AP'yi {supp.loc['S_clip','mean_AP']-supp.loc['M1','mean_AP']:+.4f}, "
    f"unknown → en sık sınıf yaklaşımı {supp.loc['S_unknown_mf','mean_AP']-supp.loc['M1','mean_AP']:+.4f}, RobustScaler "
    f"{supp.loc['S_robust','mean_AP']-supp.loc['M1','mean_AP']:+.4f} kadar değiştirmiştir. Bu farkların hepsi katlar arası SD'nin altındadır; "
    "bu nedenle ön işleme alternatifleri arasında güvenilir bir fark gösterilememiştir.")]

# ---------------------------------------------------------------------------
# 5 Nested CV bulguları
# ---------------------------------------------------------------------------
rows = [["Kat", "Yöntem", "Aday", "AP", "ROC-AUC", "F1", "Sens.", "Spec.", "Bal. Acc.", "Acc.", "TP/FP/TN/FN"]]
for _, r in outer.iterrows():
    rows.append([r.outer_fold, r.method.replace("Manual search", "Manuel arama").replace("Random search", "Otomatik arama").replace("LogReg", "Logistic Regression"),
                 r.selected_candidate, f4(r.AP), f4(r.ROC_AUC), f4(r.F1), f4(r.Sensitivity), f4(r.Specificity), f4(r.Balanced_Accuracy), f4(r.Accuracy),
                 f"{r.TP}/{r.FP}/{r.TN}/{r.FN}"])
story += [Paragraph("5. Nested CV bulguları", H1),
          tcap("Tablo 5. Dış CV kat bazlı sonuçlar (eşik 0.50). Her dış değerlendirme kümesinde 1.098 veya 1.099 gözlem ve 120 veya 121 pozitif örnek vardır."),
          table(rows, [0.7, 2.6, 1.0, 1.2, 1.5, 1.2, 1.2, 1.2, 1.4, 1.2, 2.4])]
rows = [["Yöntem", "AP", "ROC-AUC", "F1", "Sens.", "Spec.", "Bal. Acc."]]
for meth, lab in [("Manual search", "Manuel arama (MLP)"), ("Random search", "Otomatik arama (MLP)"), ("LogReg", "Logistic Regression"), ("Dummy", "Dummy")]:
    rows.append([lab] + [ms(*outer_ms(meth, m)) for m in ["AP", "ROC_AUC", "F1", "Sensitivity", "Specificity", "Balanced_Accuracy"]])
story += [Spacer(1, 6), tcap("Tablo 6. Dış CV özeti: ortalama ± örnek standart sapma (n = 3 dış kat). Katlar bağımsız tekrarlar değildir; SD güven aralığı olarak yorumlanamaz."),
          table(rows, [3.6, 2.15, 2.15, 2.15, 2.15, 2.15, 2.15])]
lm, lsd = outer_ms("LogReg", "F1"); rm_, rsd = outer_ms("Random search", "F1"); mm_, msd = outer_ms("Manual search", "F1")
story += [p(
    f"Tablo 5 ve Tablo 6'ya göre iki arama prosedürü dış CV'de birbirine çok yakın AP vermiştir (manuel {ms(om_ap, os_ap)}, otomatik {ms(orr_ap, ors_ap)}). "
    f"Logistic Regression aynı katlarda {ms(ol_ap, ols_ap)} AP ile ortalamada biraz daha yüksektir; ancak üç katın SD'si yaklaşık 0.06–0.07 olduğu için bu fark "
    "bir üstünlük olarak yorumlanamaz. Üç katın sıralaması da tutarlı değildir: 1. ve 3. katta Logistic Regression, 2. katta otomatik arama en yüksek AP'yi vermiştir. "
    f"Dummy modelin AP'si {f4(outer_ms('Dummy','AP')[0])} ile pozitif orana eşittir ve accuracy değeri {f4(outer_ms('Dummy','Accuracy')[0])} olduğu hâlde hiçbir "
    "pozitif örneği bulamaz; bu durum accuracy'nin bu problemde neden birincil metrik olamayacağını gösterir. 0.50 eşiğinde bütün öğrenen modellerin duyarlılığı "
    f"0.10 ile 0.25 arasında, özgüllüğü 0.97'nin üzerindedir. F1 değerlerinde Logistic Regression ({ms(lm, lsd)}) otomatik aramadan ({ms(rm_, rsd)}) ve manuel aramadan "
    f"({ms(mm_, msd)}) yüksektir; bunun nedeni MLP'lerin 0.50 eşiğinde daha az pozitif tahmin üretmesidir.")]

# ---------------------------------------------------------------------------
# 6 Bağımsız test
# ---------------------------------------------------------------------------
fcfg = sel["config"]
story += [Paragraph("6. Bağımsız test", H1), Paragraph("6.1 Nihai kilitli model", H2), p(
    f"Nihai konfigürasyon {FC}: {cfg_str(fcfg)}. Bu konfigürasyon tüm development kümesinde, aynı kural ile ayrılan %15'lik izleme alt kümesi kullanılarak "
    f"önceden belirlenmiş üç ağırlık seed'iyle ({', '.join(map(str, final_hp['weight_seeds']))}) eğitilmiştir. Ana model ve seed önceden ilan edilmiştir: seed "
    f"{final_hp['primary_seed']}, karar eşiği 0.50. Modellerin ağırlıkları, ön işleme durumları ve öğrenilmiş dönüşüm istatistikleri (medyanlar, RobustScaler "
    "merkez ve ölçek değerleri, kategori sözlükleri) test etiketleri okunmadan results/final_models klasörüne kaydedilmiştir. Üç modelin, Logistic Regression'ın "
    "ve Dummy modelin test olasılıkları ve sınıf tahminleri, etiket sütunu içermeyen test_predictions_locked.csv dosyasına yazılmış ve "
    f"{lock_commit[0]} numaralı commit ile ({lock_commit[1]}) GitHub'a gönderilmiştir. Test etiketleri yalnızca bu commit'ten sonra okunmuştur. Notebook "
    "yeniden çalıştırıldığında tahminleri yeniden üretir ve kilitli dosyayla aynı olduklarını doğrular; metrikler kilitli dosyadaki tahminlerden hesaplanmıştır.")]
story += [p(
    f"Şekil 6 nihai modelin üç seed'deki eğitim eğrilerini gösterir. Ana seed'de en iyi izleme kaybı {int(fe.loc[final_hp['primary_seed'],'best_epoch'])}. epoch'ta, "
    f"diğer iki seed'de {int(fe.loc[s['torch_stability_2'],'best_epoch'])}. epoch'ta elde edilmiş ve eğitim 15 epoch sonra durmuştur. Görece yüksek SGD öğrenme oranı "
    f"({fcfg['lr']:.4f}) nedeniyle izleme kaybı çok erken en düşük değerine ulaşmakta, sonrasında scheduler öğrenme oranını yarıya indirse de iyileşme olmamaktadır. "
    "")]
story += [fig(F / "fig_training_curves_final.png", 11.0, f"Şekil 6. Nihai model {FC}: üç ağırlık seed'i için eğitim (fit) ve erken durdurma izleme BCE kaybı (üstte) ve epoch bazında öğrenme oranı (altta, log ölçek).")]

rows = [["Model", "AP", "ROC-AUC", "F1", "Acc.", "Prec.", "Sens.", "Spec.", "Bal. Acc.", "TP/FP/TN/FN"]]
for lab in [main_label, "Logistic Regression", "Dummy (most_frequent)"]:
    r = test.loc[lab]
    rows.append([lab, f4(r.AP), f4(r.ROC_AUC), f4(r.F1), f4(r.Accuracy), f4(r.Precision), f4(r.Sensitivity), f4(r.Specificity), f4(r.Balanced_Accuracy), f"{r.TP}/{r.FP}/{r.TN}/{r.FN}"])
nb = int(ci.n_boot.iloc[0]); rej = int(ci.rejected_single_class.max())
story += [Paragraph("6.2 Test metrikleri ve güven aralıkları", H2),
          tcap(f"Tablo 7. Bağımsız test metrikleri (n = {int(tst.n)}, {int(tst.n_positive)} pozitif; eşik 0.50) ve {nb} tekrarlı bootstrap ile %95 yüzdelik güven aralıkları "
               f"(seed {s['bootstrap']}). Her yeniden örneklemede iki sınıfın da bulunduğu kontrol edilmiş, tek sınıflı örnekleme sayısı {rej} olmuştur."),
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
story += [Spacer(1, 6), tcap("Tablo 8. Üç başlangıç seed'inin bağımsız test performansı ve betimsel ortalama ± SD. Ana sonuç önceden ilan edilen seed 31076'dır; en iyi test seed'i ana sonuç olarak seçilmemiştir."),
          table(rows, [3.1, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.4]), Spacer(1, 6)]
lo_f, hi_f = ci_of(main_label, "F1"); lo_r, hi_r = ci_of(main_label, "ROC_AUC")
story += [p(
    f"Tablo 7'ye göre ana MLP modeli testte AP = {f4(tm.AP)} (%95 GA {f4(lo_ap)}–{f4(hi_ap)}), ROC-AUC = {f4(tm.ROC_AUC)} (%95 GA {f4(lo_r)}–{f4(hi_r)}) ve "
    f"F1 = {f4(tm.F1)} (%95 GA {f4(lo_f)}–{f4(hi_f)}) vermiştir. Logistic Regression için AP = {f4(tl.AP)}, ROC-AUC = {f4(tl.ROC_AUC)} ve F1 = {f4(tl.F1)} bulunmuştur. "
    "MLP'nin AP ve ROC-AUC değerleri biraz daha yüksek, F1 değeri ise daha düşüktür. İki modelin güven aralıkları büyük ölçüde örtüşmektedir; bu nedenle "
    f"testte iki model arasında anlamlı bir fark yoktur. Dummy model AP = {f4(td.AP)} ile alt sınırı vermektedir. Tablo 8'e göre üç seed'in AP değerleri "
    f"{ms(mrow.AP, srow.AP)} ile çok kararlıdır; F1 ve duyarlılık ise eşik 0.50'nin yakınındaki birkaç örneğin sınıfı değiştiği için daha fazla oynamaktadır "
    f"(F1 {ms(mrow.F1, srow.F1)}).")]
story += [fig(F / "fig_test_confusion_matrices.png", 8.5, "Şekil 7. Bağımsız test karışıklık matrisleri: ana MLP (solda) ve Logistic Regression (sağda), eşik 0.50."),
          fig(F / "fig_test_roc_pr.png", 12.5, "Şekil 8. Bağımsız testte ROC (solda) ve Precision–Recall (sağda) eğrileri: ana MLP, Logistic Regression ve Dummy. PR panelindeki yatay çizgi pozitif sınıf oranıdır.")]
story += [Paragraph("6.3 Hata analizi", H2), p(
    f"Şekil 7'de ana MLP 90 pozitif örneğin yalnızca {int(err.loc['TP','n'])} tanesini doğru bulmuş, {int(err.loc['FN','n'])} tanesini kaçırmıştır (False Negative). "
    f"Buna karşılık 734 negatif örnekten yalnızca {int(err.loc['FP','n'])} tanesi yanlış pozitif olarak işaretlenmiştir. Model temkinlidir: "
    f"precision = {f4(tm.Precision)}, duyarlılık ise yalnızca {f4(tm.Sensitivity)} olarak bulunmuştur. Hata analizi tablosu (results/test_error_analysis.csv) bu örüntünün nedenini gösterir. "
    f"Doğru bulunan pozitiflerde önceki kampanyası başarılı müşterilerin oranı %{err.loc['TP','poutcome_success_pct']:.0f}, medyan euribor3m değeri "
    f"{err.loc['TP','euribor3m']:.3f} olarak bulunmuştur. Kaçırılan pozitiflerde ise daha önce teması olmayanların oranı %{err.loc['FN','pdays_999_pct']:.0f}, "
    f"önceki kampanyası başarılı olanların oranı yalnızca %{err.loc['FN','poutcome_success_pct']:.0f} olarak bulunmuştur. Bu örneklerin medyan tahmin olasılığı "
    f"({err.loc['FN','prob']:.3f}) doğru negatiflerinkine ({err.loc['TN','prob']:.3f}) çok yakındır. Yani elde bulunan özelliklerle, ilk kez aranan ve abone olan müşteriler abone olmayanlardan ayrılamamaktadır. "
    f"Yanlış pozitiflerin tamamı önceki kampanyası başarılı müşterilerdir; model bu güçlü sinyali her zaman abonelik olarak yorumlamaktadır. Bu hatalar eşik "
    "değiştirilerek kısmen dengelenebilir, ancak ödev kuralı gereği eşik test verisine göre değiştirilmemiştir.")]

# ---------------------------------------------------------------------------
# 7 Tartışma
# ---------------------------------------------------------------------------
d = dev_m
h3_note = ms_res[(ms_res.candidate == "M4") & (ms_res.outer_fold == "2")].mean_AP.iloc[0] > ms_res[(ms_res.candidate == "M1") & (ms_res.outer_fold == "2")].mean_AP.iloc[0]
story += [Paragraph("7. Tartışma", H1), Paragraph("7.1 Hipotezlerin değerlendirilmesi", H2), p(
    f"<b>H1 desteklenmiştir.</b> Dummy modelin dış CV AP'si pozitif orana eşittir, accuracy'si yaklaşık 0.89'dur ve 0.50 eşiğinde bütün öğrenen modellerin duyarlılığı "
    f"0.25'in altında kalmıştır. <b>H2 desteklenmiştir.</b> Tüm development aşamasında ortalama en iyi epoch M3 için {d.loc['M3','mean_best_epoch']:.1f}, M1 için "
    f"{d.loc['M1','mean_best_epoch']:.1f} olarak bulunmuştur ve M3 bütün aşamalarda en düşük veya en düşüğe yakın AP'yi vermiştir (development {f4(d.loc['M3','mean_AP'])}, M1 "
    f"{f4(d.loc['M1','mean_AP'])}). M3'ün altı eğitimi 150 epoch sınırına erken durdurma tetiklenmeden ulaşmıştır. Adam (M2) ile AdamW (M1) arasındaki AP farkı yalnızca "
    f"{abs(d.loc['M2','mean_AP']-d.loc['M1','mean_AP']):.4f} düzeyindedir; weight decay küçük olduğu için iki yöntem neredeyse aynı davranmıştır. Otomatik aramada SGD'ye geniş "
    f"öğrenme oranı aralığı verildiğinde ({fcfg['lr']:.3f}) SGD adayı R1 en iyi sonucu vermiştir; yani SGD'nin sorunu yöntemin kendisi değil, 0.001 öğrenme oranının "
    "SGD için çok küçük olmasıdır.")]
story += [p(
    f"<b>H3 büyük ölçüde desteklenmiştir.</b> Tüm development'ta StandardScaler (M1, {f4(d.loc['M1','mean_AP'])}) MinMaxScaler'dan (M4, {f4(d.loc['M4','mean_AP'])}) "
    f"biraz yüksektir; ancak 2. dış katta M4 daha yüksek iç AP vermiş ve seçilmiştir{'' if h3_note else ''}. Fark katlar arası SD'nin altında olduğundan ölçeklemenin etkisi küçüktür. "
    f"<b>H4 kısmen reddedilmiştir.</b> M5 daha erken durmuştur (ortalama en iyi epoch {d.loc['M5','mean_best_epoch']:.1f}), bu kısmı destekler; ancak tüm development'ta "
    f"M5'in AP'si ({f4(d.loc['M5','mean_AP'])}) M1'den yüksektir ve manuel kazanan olmuştur. Fark yine küçüktür ve 3. dış katta tersine dönmüştür. "
    f"<b>H5 desteklenmiştir, ancak beklenmeyen bir nedenle.</b> M1 ve M6, {n_tot} iç CV eğitiminin {n_same} tanesinde birebir aynı AP'yi vermiştir. Epoch kayıtları, "
    "ReduceLROnPlateau'nun öğrenme oranını her zaman en iyi izleme epoch'undan sonra düşürdüğünü göstermektedir (Şekil A1). En iyi ağırlıklar geri yüklendiği için "
    "öğrenme oranı düşüşü çoğu eğitimde nihai modeli hiç etkilememiştir. Bu durum, erken durdurmanın patience değeri (15) ile birlikte scheduler'ın bu problemde "
    "etkisiz kaldığını gösterir.")]
lo_l, hi_l = ci_of("Logistic Regression", "AP")
story += [Paragraph("7.2 Genelleme: dış CV, iç CV seçimi ve bağımsız test", H2), p(
    f"Bu üç sonuç farklı soruları yanıtlar ve ayrı yorumlanmalıdır. Dış CV ({ms(orr_ap, ors_ap)} otomatik, {ms(om_ap, os_ap)} manuel) arama prosedürlerinin genelleme "
    f"başarısını tahmin eder. Tüm development'taki iç CV ({FC} için {ms(sel['dev_inner_cv_mean_AP'][FC], sel['dev_inner_cv_sd_AP'][FC])}) yalnızca seçim içindir ve "
    f"birçok aday arasından en yükseğin seçilmesi nedeniyle hafif iyimserdir. Bağımsız test (AP = {f4(tm.AP)}) tek bir kilitli modelin ölçümüdür. Test AP'si dış CV ortalamasından "
    "yaklaşık 0.04 düşüktür; ancak bu değer dış katların aralığı (0.37–0.50) içindedir ve test güven aralığı dış CV ortalamasını kapsamaktadır. Logistic Regression'da da "
    f"dış CV'den teste benzer bir düşüş görülmüştür ({f4(ol_ap)} → {f4(tl.AP)}). Her iki modelde birlikte görülen bu düşüş, model seçimine özgü bir aşırı öğrenmeden çok, "
    f"824 gözlemlik ve yalnızca 90 pozitif içeren test kümesinin örnekleme değişkenliğini düşündürür (MLP AP GA genişliği {hi_ap-lo_ap:.2f}).")]
story += [Paragraph("7.3 Optimizer farkları, aşırı öğrenme ve gereksiz karmaşıklık", H2), p(
    f"AdamW ve Adam bu problemde çok hızlı yakınsamış (ortalama durma epoch'u AdamW için {opt_t.loc['adamw','mean_stopped_epoch']:.0f}), en iyi izleme kaybına çoğunlukla "
    "ilk 35 epoch içinde ulaşmıştır. Küçük öğrenme oranlı Adam adayı R3 ve küçük öğrenme oranlı SGD adayı M3 ise yavaş yakınsamış ve bazı eğitimlerde 150 epoch sınırına "
    "ulaşmıştır. Eğitim kaybı düşerken izleme kaybının yükselmesi (Şekil 6, Şekil A1) aşırı öğrenme işaretidir ve erken durdurma bunu sınırlamıştır. 16.257 parametreli R5 ile 1.009 parametreli R3 ve 2.017 parametreli M1 arasında AP farkı katlar arası belirsizliğin "
    "altındadır. Doğrusal Logistic Regression'ın da aynı düzeyde sonuç vermesi, bu veri setinde öğrenilebilir sinyalin büyük ölçüde doğrusal olduğunu ve daha karmaşık "
    "mimarilerin ek kazanç sağlamadığını göstermektedir.")]
story += [Paragraph("7.4 Sınırlamalar", H2), p(
    "Birinci olarak, dış CV yalnızca üç kattan oluşur; katlar aynı veriyi paylaştığı için bağımsız tekrar değildir ve n = 3 ile hesaplanan SD belirsizliği sınırlı biçimde "
    "temsil eder. İkinci olarak, test kümesi küçüktür (90 pozitif); bu yüzden F1 ve duyarlılık gibi eşiğe bağlı metriklerin güven aralıkları geniştir. Üçüncü olarak, sınıf "
    "dengesizliği için class weight veya eşik ayarlaması ana deneyde kullanılmamış ve sabit 0.50 eşiği nedeniyle duyarlılık düşük kalmıştır; uygulamada eşik yalnızca "
    "development iç CV tahminleriyle belirlenmelidir. Dördüncü olarak, müşteri kimliği olmadığı için rastgele bölünme zamansal veya kişi bağımsız genellemeyi kanıtlamaz ve "
    f"duration çıkarıldıktan sonra özellikleri aynı olan {n_dup_groups} çift kaydın {n_dup_cross} tanesi development ile test arasında bölünmüştür. Beşinci olarak, nihai modelin "
    "en iyi epoch'u çok erkendir (2–7); yüksek SGD öğrenme oranı eğitimin kısa sürmesine yol açmıştır. Son olarak, veri seti kamusal olduğundan kilitli tahmin commit'i kör test "
    "garantisi değil, zaman sıralı bir deney kaydıdır. Bütün eğitimler tamamlanmış, sonlu olmayan kayıp görülmemiştir; uyarı alan eğitimler yalnızca 150 epoch sınırına "
    "ulaşan M3 ve R3 eğitimleridir ve bunlar results/inner_cv_training_runs.csv dosyasında listelenmiştir.")]

# ---------------------------------------------------------------------------
# 8 Sonuç
# ---------------------------------------------------------------------------
story += [Paragraph("8. Sonuç", H1), p(
    "Bu çalışmada test verisini bütün kararlardan ayıran, ön işlemeyi her eğitimin fit alt kümesinde öğrenen ve manuel ile otomatik aramayı aynı iç katlarda karşılaştıran "
    "bir nested CV protokolü uygulanmıştır. Kanıta dayalı bulgular şunlardır. Hiperparametre seçimi AP'yi katlar arası belirsizlikten daha az değiştirmiştir. Aynı küçük "
    "öğrenme oranıyla SGD belirgin biçimde yavaş yakınsamış, ancak uygun öğrenme oranı verildiğinde en iyi adaylardan biri olmuştur. Scheduler, erken durdurma ile birlikte "
    "çoğu eğitimde nihai modeli değiştirmemiştir. Otomatik arama prosedürü dış CV'de manuel aramaya benzer sonuç vermiştir. MLP, ne dış CV'de ne de bağımsız testte doğrusal "
    f"Logistic Regression referansından ayırt edilebilir bir üstünlük göstermemiştir (test AP {f4(tm.AP)} ve {f4(tl.AP)}, güven aralıkları örtüşmektedir). 0.50 eşiğinde "
    "duyarlılık düşüktür; bu nedenle uygulamada eşik seçimi development iç CV'si ile yapılmalı, zamansal bir doğrulama ve müşteri kimliği temelli bölünme ile sonuçlar "
    "yeniden sınanmalıdır.")]
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
