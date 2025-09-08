# -------------------------------------------------------------
#  📊 DASHBOARD FINANCIERO AVANZADO
# -------------------------------------------------------------
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import time

# Configuración global
st.set_page_config(
    page_title="📊 Dashboard Financiero Avanzado",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# Parámetros editables
Rf = 0.0435   # riesgo libre
Rm = 0.085    # retorno mercado
Tc0 = 0.21    # tasa impositiva por defecto

# Orden de sectores
SECTOR_RANK = {
    "Consumer Defensive": 1,
    "Consumer Cyclical": 2,
    "Healthcare": 3,
    "Technology": 4,
    "Financial Services": 5,
    "Industrials": 6,
    "Communication Services": 7,
    "Energy": 8,
    "Real Estate": 9,
    "Utilities": 10,
    "Basic Materials": 11,
    "Unknown": 99,
}

MAX_TICKERS_PER_CHART = 10

# =============================================================
# FUNCIONES AUXILIARES
# =============================================================
def safe_first(obj):
    if obj is None:
        return None
    if hasattr(obj, "dropna"):
        obj = obj.dropna()
    return obj.iloc[0] if hasattr(obj, "iloc") and not obj.empty else obj

def seek_row(df, keys):
    for k in keys:
        if k in df.index:
            return df.loc[k]
    return pd.Series([0], index=df.columns[:1])

def format_number(x, decimals=2, is_percent=False):
    if pd.isna(x):
        return "N/D"
    if is_percent:
        return f"{x*100:.{decimals}f}%"
    return f"{x:.{decimals}f}"

def calc_ke(beta):
    return Rf + beta * (Rm - Rf)

def calc_kd(interest, debt):
    return interest / debt if debt else 0

def calc_wacc(mcap, debt, ke, kd, t):
    total = (mcap or 0) + (debt or 0)
    return (mcap/total)*ke + (debt/total)*kd*(1-t) if total else None

def cagr4(fin, metric):
    if metric not in fin.index:
        return None
    v = fin.loc[metric].dropna().iloc[:4]
    return (v.iloc[0]/v.iloc[-1])**(1/(len(v)-1))-1 if len(v)>1 and v.iloc[-1] else None

def chunk_df(df, size=MAX_TICKERS_PER_CHART):
    if df.empty:
        return []
    return [df.iloc[i:i+size] for i in range(0, len(df), size)]

def auto_ylim(ax, values, pad=0.10):
    """Ajuste automático del eje Y."""
    if isinstance(values, pd.DataFrame):
        arr = values.to_numpy(dtype="float64")
    else:
        arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if vmax == vmin:
        ymin = vmin - abs(vmin)*pad - 1
        ymax = vmax + abs(vmax)*pad + 1
        ax.set_ylim(ymin, ymax)
        return
    if vmin >= 0:
        ymin = 0
        ymax = vmax * (1 + pad)
    elif vmax <= 0:
        ymax = 0
        ymin = vmin * (1 + pad)
    else:
        m = max(abs(vmin), abs(vmax)) * (1 + pad)
        ymin, ymax = -m, m
    ax.set_ylim(ymin, ymax)

def obtener_datos_balance_historico(ticker):
    """Obtiene datos históricos del balance para un ticker"""
    try:
        tkr = yf.Ticker(ticker)
        bs = tkr.balance_sheet
        
        if bs is None or bs.empty:
            return None, None, None, None
        
        # Obtener últimos 4 años
        bs_recent = bs.iloc[:, :4] if bs.shape[1] >= 4 else bs
        
        # Obtener datos necesarios
        activos = seek_row(bs_recent, ["Total Assets", "TotalAssets"])
        pasivos = seek_row(bs_recent, ["Total Liabilities", "TotalLiab"])
        patrimonio = seek_row(bs_recent, ["Total Stockholder Equity", "Stockholders Equity"])
        
        years = bs_recent.columns.strftime('%Y')
        
        return activos, pasivos, patrimonio, years
        
    except Exception as e:
        st.warning(f"Error obteniendo datos históricos para {ticker}: {str(e)}")
        return None, None, None, None

def obtener_datos_financieros(tk, Tc_def):
    try:
        tkr = yf.Ticker(tk)
        info = tkr.info
        bs = tkr.balance_sheet
        fin = tkr.financials
        cf = tkr.cashflow
        
        # Datos básicos
        beta = info.get("beta", 1)
        ke = calc_ke(beta)
        
        debt = safe_first(seek_row(bs, ["Total Debt", "Long Term Debt"])) or info.get("totalDebt", 0)
        cash = safe_first(seek_row(bs, [
            "Cash And Cash Equivalents",
            "Cash And Cash Equivalents At Carrying Value",
            "Cash Cash Equivalents And Short Term Investments",
        ]))
        equity = safe_first(seek_row(bs, ["Common Stock Equity", "Total Stockholder Equity"]))

        interest = safe_first(seek_row(fin, ["Interest Expense"]))
        ebt = safe_first(seek_row(fin, ["Ebt", "EBT"]))
        tax_exp = safe_first(seek_row(fin, ["Income Tax Expense"]))
        ebit = safe_first(seek_row(fin, ["EBIT", "Operating Income",
                                       "Earnings Before Interest and Taxes"]))

        kd = calc_kd(interest, debt)
        tax = tax_exp / ebt if ebt else Tc_def
        mcap = info.get("marketCap", 0)
        wacc = calc_wacc(mcap, debt, ke, kd, tax)

        nopat = ebit * (1 - tax) if ebit is not None else None
        invested = (equity or 0) + ((debt or 0) - (cash or 0))
        roic = nopat / invested if (nopat is not None and invested) else None
        
        # CALCULAR CREACIÓN DE VALOR (WACC vs ROIC) en lugar de EVA
        creacion_valor = (roic - wacc) * 100 if all(v is not None for v in (roic, wacc)) else None

        price = info.get("currentPrice")
        fcf = safe_first(seek_row(cf, ["Free Cash Flow"]))
        shares = info.get("sharesOutstanding")
        pfcf = price / (fcf/shares) if (fcf and shares) else None

        # Cálculo de ratios
        current_ratio = info.get("currentRatio")
        quick_ratio = info.get("quickRatio")
        debt_eq = info.get("debtToEquity")
        lt_debt_eq = info.get("longTermDebtToEquity")
        oper_margin = info.get("operatingMargins")
        profit_margin = info.get("profitMargins")
        roa = info.get("returnOnAssets")
        roe = info.get("returnOnEquity")
        
        # Dividendos
        div_yield = info.get("dividendYield")
        payout = info.get("payoutRatio")
        
        # Crecimiento
        revenue_growth = cagr4(fin, "Total Revenue")
        eps_growth = cagr4(fin, "Net Income")
        fcf_growth = cagr4(cf, "Free Cash Flow") or cagr4(cf, "Operating Cash Flow")

        return {
            "Ticker": tk,
            "Nombre": info.get("longName") or info.get("shortName") or info.get("displayName") or tk,
            "País": info.get("country") or info.get("countryCode") or "N/D",
            "Industria": info.get("industry") or info.get("industryKey") or info.get("industryDisp") or "N/D",
            "Sector": info.get("sector", "Unknown"),
            "Precio": price,
            "P/E": info.get("trailingPE"),
            "P/B": info.get("priceToBook"),
            "P/FCF": pfcf,
            "Dividend Yield %": div_yield,
            "Payout Ratio": payout,
            "ROA": roa,
            "ROE": roe,
            "Current Ratio": current_ratio,
            "Quick Ratio": quick_ratio,
            "Debt/Eq": debt_eq,
            "LtDebt/Eq": lt_debt_eq,
            "Oper Margin": oper_margin,
            "Profit Margin": profit_margin,
            "WACC": wacc,
            "ROIC": roic,
            "Creacion Valor (Wacc vs Roic)": creacion_valor,  # Cambiado de EVA
            "Revenue Growth": revenue_growth,
            "EPS Growth": eps_growth,
            "FCF Growth": fcf_growth,
            "MarketCap": mcap
        }
    except Exception as e:
        st.error(f"Error obteniendo datos para {tk}: {str(e)}")
        return None

# =============================================================
# INTERFAZ PRINCIPAL
# =============================================================
def main():
    st.title("📊 Dashboard de Análisis Financiero Avanzado")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        t_in = st.text_area("Tickers (separados por comas)", 
                          "HRL, AAPL, MSFT, ABT, O, XOM, KO, JNJ, CLX, CHD, CB, DDOG")
        max_t = st.slider("Máximo de tickers", 1, 100, 50)
        
        st.markdown("---")
        st.markdown("**Parámetros WACC**")
        global Rf, Rm, Tc0
        Rf = st.number_input("Tasa libre de riesgo (%)", 0.0, 20.0, 4.35)/100
        Rm = st.number_input("Retorno esperado del mercado (%)", 0.0, 30.0, 8.5)/100
        Tc0 = st.number_input("Tasa impositiva corporativa (%)", 0.0, 50.0, 21.0)/100

    if st.button("🔍 Analizar Acciones", type="primary"):
        tickers = [t.strip().upper() for t in t_in.split(",") if t.strip()][:max_t]
        
        # Obtener datos
        datos = []
        errs = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        with st.spinner("Obteniendo datos financieros..."):
            for i, tk in enumerate(tickers):
                try:
                    status_text.text(f"⏳ Procesando {tk} ({i+1}/{len(tickers)})...")
                    data = obtener_datos_financieros(tk, Tc0)
                    if data:
                        datos.append(data)
                except Exception as e:
                    errs.append({"Ticker": tk, "Error": str(e)})
                progress_bar.progress((i + 1) / len(tickers))
                time.sleep(1)  # Evitar rate limiting

        status_text.text("✅ Análisis completado!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()

        if not datos:
            st.error("No se pudieron obtener datos para los tickers proporcionados")
            if errs:
                st.table(pd.DataFrame(errs))
            return

        df = pd.DataFrame(datos)
        df["SectorRank"] = df["Sector"].map(SECTOR_RANK).fillna(99).astype(int)
        df = df.sort_values(["SectorRank", "Sector", "Ticker"])
        
        # Formatear valores para visualización
        df_disp = df.copy()
        
        # Columnas con 2 decimales
        for col in ["P/E", "P/B", "P/FCF", "Current Ratio", "Quick Ratio", "Debt/Eq", "LtDebt/Eq"]:
            df_disp[col] = df_disp[col].apply(lambda x: format_number(x, 2))
            
        # Porcentajes con 2 decimales
        for col in ["Dividend Yield %", "Payout Ratio", "ROA", "ROE", "Oper Margin", 
                   "Profit Margin", "WACC", "ROIC", "Revenue Growth", "EPS Growth", "FCF Growth"]:
            df_disp[col] = df_disp[col].apply(lambda x: format_number(x, 2, is_percent=True))
            
        # Creación de Valor con 2 decimales y porcentaje
        df_disp["Creacion Valor (Wacc vs Roic)"] = df_disp["Creacion Valor (Wacc vs Roic)"].apply(
            lambda x: format_number(x/100, 2, is_percent=True) if pd.notnull(x) else "N/D"
        )
            
        # Precio y MarketCap con 2 decimales
        df_disp["Precio"] = df_disp["Precio"].apply(lambda x: f"${float(x):,.2f}" if pd.notnull(x) else "N/D")
        df_disp["MarketCap"] = df_disp["MarketCap"].apply(lambda x: f"${float(x)/1e9:,.2f}B" if pd.notnull(x) else "N/D")
        
        # Asegurar que las columnas de texto no sean None
        for c in ["Nombre", "País", "Industria"]:
            df_disp[c] = df_disp[c].fillna("N/D").replace({None: "N/D", "": "N/D"})

        # =====================================================
        # SECCIÓN 1: RESUMEN GENERAL
        # =====================================================
        st.header("📋 Resumen General (agrupado por Sector)")
        
        # Mostrar tabla
        st.dataframe(
            df_disp[[
                "Ticker", "Nombre", "País", "Industria", "Sector",
                "Precio", "P/E", "P/B", "P/FCF",
                "Dividend Yield %", "Payout Ratio", "ROA", "ROE",
                "Current Ratio", "Debt/Eq", "Oper Margin", "Profit Margin",
                "WACC", "ROIC", "Creacion Valor (Wacc vs Roic)", "MarketCap"  # Cambiado
            ]],
            use_container_width=True,
            height=500
        )

        if errs:
            st.subheader("🚫 Tickers con error")
            st.table(pd.DataFrame(errs))

        sectors_ordered = df["Sector"].unique()

        # =====================================================
        # SECCIÓN 2: ANÁLISIS DE VALORACIÓN
        # =====================================================
        st.header("💰 Análisis de Valoración (por Sector)")
        
        for sec in sectors_ordered:
            sec_df = df[df["Sector"] == sec]
            if sec_df.empty:
                continue
                
            with st.expander(f"Sector: {sec} ({len(sec_df)} empresas)", expanded=False):
                fig, ax = plt.subplots(figsize=(10, 4))
                val = sec_df[["Ticker", "P/E", "P/B", "P/FCF"]].set_index("Ticker").apply(pd.to_numeric, errors="coerce")
                val.plot(kind="bar", ax=ax, rot=45)
                ax.set_ylabel("Ratio")
                auto_ylim(ax, val)
                st.pyplot(fig)
                plt.close()

        # =====================================================
        # SECCIÓN 3: RENTABILIDAD Y EFICIENCIA
        # =====================================================
        st.header("📈 Rentabilidad y Eficiencia")
        
        tabs = st.tabs(["ROE vs ROA", "Márgenes", "WACC vs ROIC"])

        with tabs[0]:
            for sec in sectors_ordered:
                sec_df = df[df["Sector"] == sec]
                if sec_df.empty:
                    continue
                    
                with st.expander(f"Sector: {sec}", expanded=False):
                    fig, ax = plt.subplots(figsize=(10, 5))
                    rr = pd.DataFrame({
                        "ROE": (sec_df["ROE"]*100).values,
                        "ROA": (sec_df["ROA"]*100).values
                    }, index=sec_df["Ticker"])
                    rr.plot(kind="bar", ax=ax, rot=45)
                    ax.set_ylabel("%")
                    auto_ylim(ax, rr)
                    st.pyplot(fig)
                    plt.close()

        with tabs[1]:
            for sec in sectors_ordered:
                sec_df = df[df["Sector"] == sec]
                if sec_df.empty:
                    continue
                    
                with st.expander(f"Sector: {sec}", expanded=False):
                    fig, ax = plt.subplots(figsize=(10, 5))
                    mm = pd.DataFrame({
                        "Oper Margin": (sec_df["Oper Margin"]*100).values,
                        "Profit Margin": (sec_df["Profit Margin"]*100).values
                    }, index=sec_df["Ticker"])
                    mm.plot(kind="bar", ax=ax, rot=45)
                    ax.set_ylabel("%")
                    auto_ylim(ax, mm)
                    st.pyplot(fig)
                    plt.close()

        with tabs[2]:
            fig, ax = plt.subplots(figsize=(12, 6))
            rw = pd.DataFrame({
                "ROIC": (df["ROIC"]*100).values,
                "WACC": (df["WACC"]*100).values
            }, index=df["Ticker"])
            rw.plot(kind="bar", ax=ax, rot=45)
            ax.set_ylabel("%")
            ax.set_title("Creación de Valor: ROIC vs WACC")
            auto_ylim(ax, rw)
            st.pyplot(fig)
            plt.close()

        # =============================================================
        # SECCIÓN 4: ESTRUCTURA DE CAPITAL Y LIQUIDEZ
        # =============================================================
        st.header("🏦 Estructura de Capital y Liquidez (por sector)")

        for sec in sectors_ordered:
            sec_df = df[df["Sector"] == sec]
            if sec_df.empty:
                continue
                
            with st.expander(f"Sector: {sec}", expanded=False):
                for i, chunk in enumerate(chunk_df(sec_df), 1):
                    st.caption(f"Bloque {i}")
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        st.caption("Evolución Patrimonio Deuda Activos (últimos 4 años)")
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        has_data = False
                        
                        for ticker in chunk["Ticker"]:
                            activos, pasivos, patrimonio, years = obtener_datos_balance_historico(ticker)
                            
                            if activos is not None and not activos.empty:
                                has_data = True
                                # Crear DataFrame para el gráfico
                                df_evol = pd.DataFrame({
                                    'Activos': activos.values / 1e6,  # Convertir a millones
                                    'Pasivos': pasivos.values / 1e6,
                                    'Patrimonio': patrimonio.values / 1e6
                                }, index=years)
                                
                                # Graficar
                                x_pos = np.arange(len(years))
                                width = 0.25
                                
                                ax.bar(x_pos - width, df_evol['Activos'], width, 
                                      label=f'{ticker} - Activos', alpha=0.8)
                                ax.bar(x_pos, df_evol['Pasivos'], width, 
                                      label=f'{ticker} - Pasivos', alpha=0.8)
                                ax.bar(x_pos + width, df_evol['Patrimonio'], width, 
                                      label=f'{ticker} - Patrimonio', alpha=0.8)
                        
                        if has_data:
                            ax.set_xlabel('Año')
                            ax.set_ylabel('Valor (Millones USD)')
                            ax.set_title('Evolución Patrimonio Deuda Activos')
                            ax.set_xticks(np.arange(len(years)))
                            ax.set_xticklabels(years)
                            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                            plt.xticks(rotation=45)
                            st.pyplot(fig)
                        else:
                            st.info(f"No se encontraron datos históricos de balance para las empresas del sector {sec}")
                        plt.close()
                        
                    with c2:
                        st.caption("Liquidez")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        liq = chunk[["Ticker", "Current Ratio", "Quick Ratio"]].set_index("Ticker").apply(pd.to_numeric, errors="coerce")
                        liq.plot(kind="bar", ax=ax, rot=45)
                        ax.axhline(1, color="green", linestyle="--")
                        ax.set_ylabel("Ratio")
                        auto_ylim(ax, liq)
                        st.pyplot(fig)
                        plt.close()

        # =====================================================
        # SECCIÓN 5: CRECIMIENTO
        # =====================================================
        st.header("🚀 Crecimiento (CAGR 3-4 años, por sector)")
        
        for sec in sectors_ordered:
            sec_df = df[df["Sector"] == sec]
            if sec_df.empty:
                continue
                
            with st.expander(f"Sector: {sec}", expanded=False):
                for i, chunk in enumerate(chunk_df(sec_df), 1):
                    st.caption(f"Bloque {i}")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    gdf = pd.DataFrame({
                        "Revenue Growth": (chunk["Revenue Growth"]*100).values,
                        "EPS Growth": (chunk["EPS Growth"]*100).values,
                        "FCF Growth": (chunk["FCF Growth"]*100).values
                    }, index=chunk["Ticker"])
                    gdf.plot(kind="bar", ax=ax, rot=45)
                    ax.axhline(0, color="black", linewidth=0.8)
                    ax.set_ylabel("%")
                    auto_ylim(ax, gdf)
                    st.pyplot(fig)
                    plt.close()

        # =====================================================
        # SECCIÓN 6: ANÁLISIS INDIVIDUAL
        # =====================================================
        st.header("🔍 Análisis por Empresa")
        pick = st.selectbox("Selecciona empresa", df_disp["Ticker"].unique())
        det_disp = df_disp[df_disp["Ticker"] == pick].iloc[0]
        det_raw = df[df["Ticker"] == pick].iloc[0]

        st.markdown(f"""
        **{det_raw['Nombre']}**  
        **Sector:** {det_raw['Sector']}  
        **País:** {det_raw['País']}  
        **Industria:** {det_raw['Industria']}
        """)

        cA, cB, cC = st.columns(3)
        with cA:
            st.metric("Precio", det_disp["Precio"])
            st.metric("P/E", det_disp["P/E"])
            st.metric("P/B", det_disp["P/B"])
            st.metric("P/FCF", det_disp["P/FCF"])
            
        with cB:
            st.metric("Market Cap", det_disp["MarketCap"])
            st.metric("ROIC", det_disp["ROIC"])
            st.metric("WACC", det_disp["WACC"])
            st.metric("Creación Valor", det_disp["Creacion Valor (Wacc vs Roic)"])  # Cambiado
            
        with cC:
            st.metric("ROE", det_disp["ROE"])
            st.metric("Dividend Yield", det_disp["Dividend Yield %"])
            st.metric("Current Ratio", det_disp["Current Ratio"])
            st.metric("Debt/Eq", det_disp["Debt/Eq"])

        st.subheader("ROIC vs WACC")
        if pd.notnull(det_raw["ROIC"]) and pd.notnull(det_raw["WACC"]):
            fig, ax = plt.subplots(figsize=(5, 4))
            comp = pd.DataFrame({
                "ROIC": [det_raw["ROIC"]*100],
                "WACC": [det_raw["WACC"]*100]
            }, index=[pick])
            comp.plot(kind="bar", ax=ax, rot=0, legend=False, 
                     color=["green" if det_raw["ROIC"] > det_raw["WACC"] else "red", "gray"])
            ax.set_ylabel("%")
            auto_ylim(ax, comp)
            st.pyplot(fig)
            plt.close()
            
            if det_raw["ROIC"] > det_raw["WACC"]:
                st.success("✅ Crea valor (ROIC > WACC)")
            else:
                st.error("❌ Destruye valor (ROIC < WACC)")
        else:
            st.warning("Datos insuficientes para comparar ROIC/WACC")

if __name__ == "__main__":
    main()
