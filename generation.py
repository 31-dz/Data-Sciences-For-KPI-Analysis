import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define flare causes and their typical characteristics
# Flare rates in cubic meters per hour (m³/hr)
FLARE_CAUSES = {
    'normal_operations': {'baseline': 100, 'std': 8},
    'process_upset': {'avg_rate': 50, 'std': 15, 'probability': 0.15},
    'equipment_maintenance': {'avg_rate': 40, 'std': 12, 'probability': 0.10},
    'startup_shutdown': {'avg_rate': 200, 'std': 50, 'probability': 0.00},  # Scheduled
    'emergency_relief': {'avg_rate': 80, 'std': 25, 'probability': 0.03},
    'compressor_trip': {'avg_rate': 120, 'std': 30, 'probability': 0.02},
    'instrument_failure': {'avg_rate': 60, 'std': 20, 'probability': 0.02}
}

def generate_flare_data(year=2024):
    """Generate hourly flare gas data with multi-cause contributions"""
    
    # Create datetime range for the year
    start_date = datetime(year, 1, 1, 0, 0, 0)
    end_date = datetime(year, 12, 31, 23, 0, 0)
    date_range = pd.date_range(start=start_date, end=end_date, freq='h')
    
    # Schedule exactly 4 startup/shutdown events throughout the year
    shutdown_events = []
    shutdown_dates = [
        datetime(year, 2, 15),   # Mid-February
        datetime(year, 4, 20),   # Spring turnaround
        datetime(year, 7, 10),   # Summer maintenance
        datetime(year, 10, 5)    # Fall preparation
    ]
    
    for shutdown_date in shutdown_dates:
        duration_hours = random.randint(24, 72)
        shutdown_start = shutdown_date + timedelta(hours=random.randint(0, 23))
        for h in range(duration_hours):
            shutdown_events.append(shutdown_start + timedelta(hours=h))
    
    data = []
    
    # Initialize baseline for smooth transitions
    baseline_rate = 100
    
    for i, dt in enumerate(date_range):
        # Check if current time is during a scheduled shutdown
        is_shutdown = dt in shutdown_events
        
        # Initialize contributions for each cause
        contributions = {
            'normal_operations': 0,
            'process_upset': 0,
            'equipment_maintenance': 0,
            'startup_shutdown': 0,
            'emergency_relief': 0,
            'compressor_trip': 0,
            'instrument_failure': 0
        }
        
        # Normal operations - always present as baseline
        baseline_rate += np.random.normal(0, 0.5)
        baseline_rate = np.clip(baseline_rate, 80, 120)
        contributions['normal_operations'] = max(0, baseline_rate + np.random.normal(0, FLARE_CAUSES['normal_operations']['std']))
        
        # Add time-based variation to baseline
        hour = dt.hour
        if 0 <= hour < 6:
            contributions['normal_operations'] *= 0.95
        elif 8 <= hour < 16:
            contributions['normal_operations'] *= 1.02
        
        # Seasonal variation
        month = dt.month
        if 6 <= month <= 8:
            contributions['normal_operations'] *= 1.05
        
        # Handle shutdown events
        if is_shutdown:
            shutdown_rate = np.random.normal(
                FLARE_CAUSES['startup_shutdown']['avg_rate'],
                FLARE_CAUSES['startup_shutdown']['std']
            )
            contributions['startup_shutdown'] = max(0, shutdown_rate)
        
        # Other causes occur probabilistically (not during shutdowns)
        if not is_shutdown:
            for cause in ['process_upset', 'equipment_maintenance', 'emergency_relief', 
                         'compressor_trip', 'instrument_failure']:
                if random.random() < FLARE_CAUSES[cause]['probability']:
                    rate = np.random.normal(
                        FLARE_CAUSES[cause]['avg_rate'],
                        FLARE_CAUSES[cause]['std']
                    )
                    contributions[cause] = max(0, rate)
        
        # Calculate total flare rate
        total_flare_rate = sum(contributions.values())
        
        # Determine severity and dominant cause
        severity = 'low'
        if total_flare_rate > 350:
            severity = 'high'
        elif total_flare_rate > 200:
            severity = 'medium'
        
        # Find dominant cause
        dominant_cause = max(contributions.items(), key=lambda x: x[1])[0]
        
        data.append({
            'timestamp': dt,
            'total_flare_rate_m3_per_hour': round(total_flare_rate, 2),
            'normal_operations_m3_per_hour': round(contributions['normal_operations'], 2),
            'process_upset_m3_per_hour': round(contributions['process_upset'], 2),
            'equipment_maintenance_m3_per_hour': round(contributions['equipment_maintenance'], 2),
            'startup_shutdown_m3_per_hour': round(contributions['startup_shutdown'], 2),
            'emergency_relief_m3_per_hour': round(contributions['emergency_relief'], 2),
            'compressor_trip_m3_per_hour': round(contributions['compressor_trip'], 2),
            'instrument_failure_m3_per_hour': round(contributions['instrument_failure'], 2),
            'dominant_cause': dominant_cause,
            'severity': severity,
            'day_of_week': dt.strftime('%A'),
            'month': dt.strftime('%B'),
            'hour': dt.hour
        })
    
    return pd.DataFrame(data)

def generate_summary_statistics(df):
    """Generate summary statistics for the flare data"""
    
    cause_columns = [
        'normal_operations_m3_per_hour',
        'process_upset_m3_per_hour',
        'equipment_maintenance_m3_per_hour',
        'startup_shutdown_m3_per_hour',
        'emergency_relief_m3_per_hour',
        'compressor_trip_m3_per_hour',
        'instrument_failure_m3_per_hour'
    ]
    
    print("=" * 70)
    print("LNG FLARE GAS ANNUAL SUMMARY")
    print("=" * 70)
    print(f"\nTotal Hours of Operation: {len(df):,}")
    print(f"Total Flare Gas (m³): {df['total_flare_rate_m3_per_hour'].sum():,.2f}")
    print(f"Average Hourly Rate (m³/hr): {df['total_flare_rate_m3_per_hour'].mean():,.2f}")
    print(f"Peak Rate (m³/hr): {df['total_flare_rate_m3_per_hour'].max():,.2f}")
    print(f"Minimum Rate (m³/hr): {df['total_flare_rate_m3_per_hour'].min():,.2f}")
    
    print("\n" + "=" * 70)
    print("FLARE BY CAUSE - TOTAL CONTRIBUTION (m³)")
    print("=" * 70)
    for col in cause_columns:
        cause_name = col.replace('_m3_per_hour', '').replace('_', ' ').title()
        total = df[col].sum()
        percentage = (total / df['total_flare_rate_m3_per_hour'].sum()) * 100
        avg = df[col].mean()
        max_val = df[col].max()
        events = (df[col] > 0).sum()
        print(f"{cause_name:30s}: {total:>12,.2f} ({percentage:5.2f}%) | Avg: {avg:>6.2f} | Max: {max_val:>6.2f} | Events: {events:>5,}")
    
    print("\n" + "=" * 70)
    print("FLARE BY SEVERITY")
    print("=" * 70)
    severity_summary = df.groupby('severity').agg({
        'total_flare_rate_m3_per_hour': ['count', 'sum', 'mean']
    }).round(2)
    print(severity_summary)
    
    print("\n" + "=" * 70)
    print("DOMINANT CAUSE DISTRIBUTION")
    print("=" * 70)
    dominant_counts = df['dominant_cause'].value_counts()
    print(dominant_counts)
    
    print("\n" + "=" * 70)
    print("MONTHLY TOTALS (m³)")
    print("=" * 70)
    monthly = df.groupby('month')['total_flare_rate_m3_per_hour'].sum().round(2)
    print(monthly)

# Generate the data
print("Generating flare gas data for 2024...")
print("Scheduled shutdown events:")
print("  1. Mid-February (Feb 15)")
print("  2. Spring Turnaround (Apr 20)")
print("  3. Summer Maintenance (Jul 10)")
print("  4. Fall Preparation (Oct 5)")
print()
flare_df = generate_flare_data(2024)

# Display summary statistics
generate_summary_statistics(flare_df)

# Save to CSV
output_file = 'lng_flare_data.csv'
flare_df.to_csv(output_file, index=False)
print(f"\n✓ Data saved to '{output_file}'")

# Display sample of the data
print("\n" + "=" * 70)
print("SAMPLE DATA (First 10 rows)")
print("=" * 70)
print(flare_df.head(10).to_string(index=False))

print("\n" + "=" * 70)
print("HIGH SEVERITY EVENTS (Sample)")
print("=" * 70)
high_severity = flare_df[flare_df['severity'] == 'high'].head(5)
print(high_severity[['timestamp', 'total_flare_rate_m3_per_hour', 'startup_shutdown_m3_per_hour', 
                      'compressor_trip_m3_per_hour', 'emergency_relief_m3_per_hour', 
                      'dominant_cause', 'severity']].to_string(index=False))