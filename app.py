import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from itertools import permutations
from matplotlib.patches import Circle
from kneed import KneeLocator

st.set_page_config(page_title="Smart Irrigation Advanced", layout="wide")
st.title("🌱 إدارة الري الذكي مع مسار الدرون")

# ===== 1) إعدادات الحقل =====
st.sidebar.header("إعدادات الحقل")
FIELD_SIZE = st.sidebar.number_input("مساحة الحقل (m²)", min_value=10, value=100)
NUM_SENSORS = st.sidebar.number_input("عدد الحساسات", min_value=5, value=50)
TX_RANGE = st.sidebar.number_input("مدى الإرسال (m)", min_value=5, value=25)

# ===== 2) رفع ملف CSV =====
st.sidebar.header("رفع ملف بيانات الطقس/الزراعة")
uploaded_file = st.sidebar.file_uploader("اختر ملف CSV", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success(f"✅ تم تحميل البيانات! شكل البيانات: {data.shape}")

    required_cols = ["temperature", "humidity", "rainfall"]
    if not all(col in data.columns for col in required_cols):
        st.error(f"الملف يجب أن يحتوي على الأعمدة: {required_cols}")
    else:
        # ===== 3) معالجة البيانات =====
        df = data[required_cols].copy()
        df['moisture'] = 0.5 * df['humidity'] + 0.5 * df['rainfall']
        eps = 1e-8
        df['Ir'] = df['temperature'] / (df['moisture'] + eps)
        st.write("✅ تم معالجة البيانات وحساب moisture و Ir")
        st.dataframe(df.head())

        # ===== 4) توزيع الحساسات =====
        sensor_positions = np.random.rand(NUM_SENSORS, 2) * FIELD_SIZE

        # ===== 5) تحديد CH باستخدام KMeans =====
        K_max = max(3, min(NUM_SENSORS // 2, 20))
        sse = []
        for k in range(2, K_max + 1):
            km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(sensor_positions)
            sse.append(km.inertia_)
        kl = KneeLocator(range(2, K_max + 1), sse, curve="convex", direction="decreasing")
        best_k = kl.knee if kl.knee else 2
        st.write(f"✅ أفضل عدد أولي للعناقيد (CHs) = {best_k}")

        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto").fit(sensor_positions)
        CH_positions = kmeans.cluster_centers_

        # ===== 6) التحقق من شرط TX_RANGE =====
        final_CHs = CH_positions.tolist()
        sensor_to_CH = []
        for sensor in sensor_positions:
            distances = np.linalg.norm(sensor - np.array(final_CHs), axis=1)
            nearest_CH_idx = np.argmin(distances)
            if distances[nearest_CH_idx] <= TX_RANGE:
                sensor_to_CH.append(nearest_CH_idx)
            else:
                final_CHs.append(sensor.tolist())
                sensor_to_CH.append(len(final_CHs)-1)
        final_CHs = np.array(final_CHs)
        st.write(f"ℹ️ العدد النهائي لمراكز العناقيد (CHs) بعد شرط المدى: {len(final_CHs)}")

        # ===== 7) رسم توزيع الحساسات وCHs =====
        fig, ax = plt.subplots(figsize=(8,8))
        clusters = np.array(sensor_to_CH)
        ax.scatter(sensor_positions[:,0], sensor_positions[:,1], c=clusters, cmap='tab20', alpha=0.6, label='Sensors')
        ax.scatter(final_CHs[:,0], final_CHs[:,1], c='green', s=120, marker='X', label='CHs')
        for ch in final_CHs:
            circle = Circle((ch[0], ch[1]), TX_RANGE, color='green', alpha=0.1)
            ax.add_patch(circle)
        ax.set_title("توزيع الحساسات ورؤوس العناقيد (CHs)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend()
        st.pyplot(fig)

        # ===== 8) مسار الدرون باستخدام TSP =====
        def tsp_brute_force(points):
            n = len(points)
            min_dist = float('inf')
            best_path = None
            for perm in permutations(range(n)):
                dist = sum(np.linalg.norm(points[perm[i]] - points[perm[i+1]]) for i in range(n-1))
                dist += np.linalg.norm(points[perm[-1]] - points[perm[0]])
                if dist < min_dist:
                    min_dist = dist
                    best_path = perm
            return best_path, min_dist

        tsp_path, tsp_len = tsp_brute_force(final_CHs)
        st.write(f"🚁 طول مسار الدرون المثالي (TSP) = {tsp_len:.2f} m")
        st.write("مسار الدرون (ترتيب CHs):", tsp_path)

        # ===== 9) تحميل نموذج الري =====
        best_model = joblib.load("best_ir_model.joblib")
        st.success("✅ تم تحميل نموذج التنبؤ بالري")

        # ===== 10) تجميع البيانات لكل CH =====
        df_exp = df.copy()
        n_ch = len(final_CHs)
        df_exp["CH_id"] = df_exp.index % n_ch
        num_cols = df_exp.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy="mean")
        df_exp[num_cols] = imputer.fit_transform(df_exp[num_cols])

        ch_agg = df_exp.groupby("CH_id").mean().reset_index()
        X_agg = ch_agg[required_cols + ['moisture']]
        y_pred = best_model.predict(X_agg)
        ch_agg['Predicted_Ir'] = y_pred

        st.subheader("📊 تنبؤ الري لكل CH")
        st.dataframe(ch_agg[['CH_id', 'Predicted_Ir']])

        # ===== 11) رسم مسار الدرون الديناميكي + Base Station + جدول دوري =====
        BS = np.array([[FIELD_SIZE/2, FIELD_SIZE + 10]])  # Base Station
        drone_tour_order = list(tsp_path) + ["BS"]
        # ===== Sidebar TDMA Settings =====
st.sidebar.header("تفاعلية مسار الدرون مع TDMA")
cycle_duration = st.sidebar.number_input("مدة الدورة الزمنية لكل دورة (دقيقة)", min_value=1, value=20)
total_hours = st.sidebar.number_input("إجمالي ساعات التشغيل", min_value=1, value=3)
total_cycles = (total_hours * 60) // cycle_duration
if total_cycles == 0:
    total_cycles = 1
st.sidebar.write(f"عدد الدورات الكلي = {total_cycles}")

# ===== Slider لاختيار الدورة الحالية =====
cycle_idx = st.sidebar.slider("اختر الدورة الزمنية (Cycle)", 1, total_cycles, 1)

# ===== حساب عدد CHs لكل دورة =====
ch_per_cycle = len(tsp_path) // total_cycles
if ch_per_cycle == 0:
    ch_per_cycle = 1

# تحديد CHs التي سيتم زيارتها في الدورة الحالية
end_idx = min(cycle_idx * ch_per_cycle, len(tsp_path))
current_CHs = np.array(tsp_path[:end_idx])

# ===== رسم المسار الديناميكي مع TDMA =====
fig, ax = plt.subplots(figsize=(8,8))
clusters = np.array(sensor_to_CH)
ax.scatter(sensor_positions[:,0], sensor_positions[:,1], c=clusters, cmap='tab20', alpha=0.6, s=80, label='Sensors')
ax.scatter(final_CHs[:,0], final_CHs[:,1], c='green', s=120, marker='X', edgecolor='black', label='CHs')
ax.scatter(BS[0,0], BS[0,1], c='red', s=150, marker='*', label='Base Station')

# تلوين CHs حسب IR إذا تم زيارتها حتى الدورة الحالية
colors = []
for idx in range(len(final_CHs)):
    if idx in current_CHs:
        ir = ch_agg.loc[ch_agg['CH_id'] == idx, 'Predicted_Ir'].values[0]
        if ir > 2.0:
            colors.append('red')
        elif ir < 1.0:
            colors.append('green')
        else:
            colors.append('yellow')
    else:
        colors.append('gray')
ax.scatter(final_CHs[:,0], final_CHs[:,1], c=colors, s=120, marker='X', label='CHs')

# رسم مسار الدرون حتى الدورة الحالية + العودة إلى BS
if len(current_CHs) > 0:
    path_points = final_CHs[current_CHs]
    path_points = np.vstack([path_points, BS])
    ax.plot(path_points[:,0], path_points[:,1], c='black', linestyle='-', marker='o', label='Drone Path')
    
    # اتصال آخر CH بالـ BS
    last_ch = final_CHs[current_CHs[-1]]
    ax.plot([last_ch[0], BS[0,0]], [last_ch[1], BS[0,1]], c='blue', linestyle='--', linewidth=2, label='Drone → BS')

# دائرة TX_RANGE لكل CH
for ch in final_CHs:
    circle = Circle((ch[0], ch[1]), TX_RANGE, color='green', alpha=0.1)
    ax.add_patch(circle)

ax.set_title(f"Cycle {cycle_idx}/{total_cycles} - Drone Tour with TDMA & CH IR")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.legend(loc='upper right')
ax.grid(True)
ax.axis('equal')
st.pyplot(fig)

# ===== جدول دوري لمسار الدرون =====
tour_table = []
for i, stop in enumerate(list(current_CHs) + ["BS"]):
    if stop == "BS":
        tour_table.append({"Step": i+1, "Visited": "Base Station"})
    else:
        ir = ch_agg.loc[ch_agg['CH_id'] == stop, 'Predicted_Ir'].values[0]
        tour_table.append({"Step": i+1, "Visited": f"CH {stop}", "Predicted_Ir": round(ir, 2)})

st.subheader("📋 جدول دوري لمسار الدرون")
st.table(pd.DataFrame(tour_table))

       

               

       # ===== 11) تنبيهات الري (تم تعديل الحدود) =====
st.subheader("⚠️ تنبيهات الري")
for i, row in ch_agg.iterrows():
    ir = row['Predicted_Ir']
    if ir > 1.5:
        msg = "❌ رطوبة منخفضة — يلزم الري الفوري"
    elif 0.5 <= ir <= 1.5:
        msg = "⚠️ رطوبة معتدلة — راقب الحقل"
    else:  # ir < 0.5
        msg = "✅ رطوبة مناسبة"
    st.write(f"CH {row['CH_id']}: {ir:.2f} → {msg}")

