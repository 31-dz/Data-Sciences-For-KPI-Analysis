import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for better-looking plots
sns.set_style("whitegrid")

# Load the data
print("Loading flare gas data...")
df = pd.read_csv('lng_flare_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 1. Time Series - Full Year Overview
plt.figure(figsize=(15, 6))
plt.plot(df['timestamp'], df['total_flare_rate_m3_per_hour'], linewidth=0.5, alpha=0.7, color='#e74c3c')
plt.title('Flare Gas Rate - Full Year 2024', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Flare Rate (m³/hr)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot1_full_year_overview.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Daily Average Flare Rate
plt.figure(figsize=(15, 6))
daily_avg = df.groupby(df['timestamp'].dt.date)['total_flare_rate_m3_per_hour'].mean()
plt.plot(daily_avg.index, daily_avg.values, linewidth=2, color='#3498db')
plt.fill_between(daily_avg.index, daily_avg.values, alpha=0.3, color='#3498db')
plt.title('Daily Average Flare Rate', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Avg Flare Rate (m³/hr)')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plot2_daily_average.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Flare Rate by Hour of Day
plt.figure(figsize=(12, 6))
hourly_avg = df.groupby('hour')['total_flare_rate_m3_per_hour'].mean()
plt.bar(hourly_avg.index, hourly_avg.values, color='#2ecc71', alpha=0.7, edgecolor='black')
plt.title('Average Flare Rate by Hour of Day', fontsize=14, fontweight='bold')
plt.xlabel('Hour of Day')
plt.ylabel('Avg Flare Rate (m³/hr)')
plt.xticks(range(0, 24, 2))
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('plot3_hourly_pattern.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Monthly Total Flare Gas
plt.figure(figsize=(12, 6))
df['month_num'] = df['timestamp'].dt.month
monthly_total = df.groupby('month_num')['total_flare_rate_m3_per_hour'].sum() / 1000  # Convert to thousands
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
colors = plt.cm.viridis(np.linspace(0, 1, 12))
plt.bar(range(1, 13), monthly_total.values, color=colors, edgecolor='black')
plt.title('Monthly Total Flare Gas', fontsize=14, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Total Flare Gas (1000 m³)')
plt.xticks(range(1, 13), month_names)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('plot4_monthly_totals.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Flare Rate by Cause (using individual cause columns)
plt.figure(figsize=(12, 7))
cause_columns = [
    'normal_operations_m3_per_hour',
    'process_upset_m3_per_hour',
    'equipment_maintenance_m3_per_hour',
    'startup_shutdown_m3_per_hour',
    'emergency_relief_m3_per_hour',
    'compressor_trip_m3_per_hour',
    'instrument_failure_m3_per_hour'
]

cause_totals = {col.replace('_m3_per_hour', ''): df[col].sum() for col in cause_columns}
cause_data = pd.Series(cause_totals).sort_values(ascending=True)

cause_colors = {
    'normal_operations': '#95a5a6', 
    'process_upset': '#e67e22', 
    'equipment_maintenance': '#9b59b6', 
    'startup_shutdown': '#f39c12',
    'emergency_relief': '#e74c3c', 
    'compressor_trip': '#c0392b',
    'instrument_failure': '#d35400'
}

colors_list = [cause_colors.get(cause, '#34495e') for cause in cause_data.index]
plt.barh(range(len(cause_data)), cause_data.values / 1000, color=colors_list, edgecolor='black')
plt.yticks(range(len(cause_data)), [c.replace('_', ' ').title() for c in cause_data.index])
plt.title('Total Flare Gas by Cause', fontsize=14, fontweight='bold')
plt.xlabel('Total Flare Gas (1000 m³)')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('plot5_flare_by_cause.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Stacked Area Chart showing contribution by cause over time
plt.figure(figsize=(15, 7))
# Resample to daily for cleaner visualization
daily_data = df.set_index('timestamp')[cause_columns].resample('D').sum()

plt.stackplot(daily_data.index, 
              daily_data['normal_operations_m3_per_hour'],
              daily_data['process_upset_m3_per_hour'],
              daily_data['equipment_maintenance_m3_per_hour'],
              daily_data['startup_shutdown_m3_per_hour'],
              daily_data['emergency_relief_m3_per_hour'],
              daily_data['compressor_trip_m3_per_hour'],
              daily_data['instrument_failure_m3_per_hour'],
              labels=['Normal Operations', 'Process Upset', 'Equipment Maintenance', 'Startup/Shutdown', 'Emergency Relief', 'Compressor Trip', 'Instrument Failure'],
              colors=['#95a5a6', '#e67e22', '#9b59b6', '#f39c12', '#e74c3c', '#c0392b', '#d35400'],
              alpha=0.8)

plt.title('Daily Flare Gas Contribution by Cause', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Daily Total Flare Gas (m³)')
plt.legend(loc='upper left', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot6_stacked_causes.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. BONUS: Severity Distribution
plt.figure(figsize=(10, 6))
severity_counts = df['severity'].value_counts()
severity_colors = {'low': '#2ecc71', 'medium': '#f39c12', 'high': '#e74c3c'}
colors_severity = [severity_colors[sev] for sev in severity_counts.index]

plt.bar(severity_counts.index, severity_counts.values, color=colors_severity, edgecolor='black', alpha=0.8)
plt.title('Distribution of Flare Events by Severity', fontsize=14, fontweight='bold')
plt.xlabel('Severity Level')
plt.ylabel('Number of Hours')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('plot7_severity_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 6))

# Filter out zero values for better visualization
normal_ops_data = df[df['normal_operations_m3_per_hour'] > 0]['normal_operations_m3_per_hour']

# 8. Create histogram with KDE overlay
plt.hist(normal_ops_data, bins=50, color='#3498db', alpha=0.6, edgecolor='black', density=True, label='Histogram')
normal_ops_data.plot(kind='kde', color='#e74c3c', linewidth=2, label='KDE')

plt.title('Distribution of Normal Operations Flare Rate', fontsize=14, fontweight='bold')
plt.xlabel('Flare Rate (m³/hr)')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Add statistics text box
stats_text = f'Mean: {normal_ops_data.mean():.1f} m³/hr\nMedian: {normal_ops_data.median():.1f} m³/hr\nStd Dev: {normal_ops_data.std():.1f} m³/hr'
plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes, 
         fontsize=10, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('plot8_normal_ops_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nAll plots have been generated and saved successfully!")
print("Total plots created: 8")

