import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects
import streamlit as st

# 1. Load Cambodia GeoJSON (admin level 1 = province)
cambodia = gpd.read_file("/Users/thekhemfee/Downloads/Intelligent_Loan_Approval/frontend/geoBoundaries-KHM-ADM1_simplified.geojson")

# 2. Loan amount by province
data = pd.DataFrame({
    'Province': [
        'Bantey Meanchey',
        'Phnom Penh',
        'Takeo',
        'Kampong Chhnang',
        'Tbong Khmum',
        'Kampot',
        'Kandal',
        'Siem Reap',
        'Kampong Speu',
        'Prey Veng',
        'Kampong Thom',
        'Pursat',
        'Stung Treng',
        'Preah Sihanouk',
        'Battambang',
        'Kampong Cham',
        'Kep',
        'Koh Kong',
        'Kratie',
        'Mondulkiri',
        'Oddar Meanchey',
        'Pailin',
        'Preah Vihear',
        'Prey Veng',
        'Svay Rieng',
        'Stung Treng',
        'Ratanakiri Province' 

    ],
    'LoanAmount': [
        289000.00,
        212057.00,
        183000.00,
        140000.00,
        100000.00,
        73013.50,
        55000.00,
        45000.00,
        5000.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00
    ]
})

# 3. Merge geospatial + disbursement data
merged = cambodia.merge(data,
                        left_on='shapeName',   # name of province in GeoJSON
                        right_on='Province',
                        how='left')

# 4. Plot map with fill color based on LoanAmount
fig, ax = plt.subplots(figsize=(10, 8))
merged.plot(column='LoanAmount',
            ax=ax,
            legend=True,
            cmap='YlOrRd',
            missing_kwds={'color': 'lightgrey', 'label': 'No Data'},
            edgecolor='black')

plt.title("CMCB Outlet Loan Disbursement by Province ")
plt.axis('off')

# 5. Optional: add province labels with value
for idx, row in merged.iterrows():
    if pd.notna(row['LoanAmount']):
        x, y = row['geometry'].centroid.coords[0]
        
        # Draw white outline first (stroke effect)
        plt.text(
            x, y,
            f"{row['shapeName']}\n${int(row['LoanAmount']):,}",
            ha='center',
            fontsize=9,
            color='white',
            weight='bold',
            path_effects=[
                matplotlib.patheffects.withStroke(linewidth=3, foreground="white")
            ],
            zorder=3
        )
        
        # Draw actual label on top
        plt.text(
            x, y,
            f"{row['shapeName']}\n${int(row['LoanAmount']):,}",
            ha='center',
            fontsize=6,
            color='black',
            weight='bold',
            zorder=4
        )

# 6. Show in Streamlit
st.pyplot(fig)