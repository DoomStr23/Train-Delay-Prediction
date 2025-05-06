import pandas as pd
import numpy as np
from datetime import datetime
import re

def clean_train_data(input_file="train_data_log.csv", output_file="clean_train_data.csv"):
    """
    Clean and process Deutsche Bahn train data, adding delay and other useful information.
    Data is sorted chronologically by date and time.
    Duplicates are removed based on station_eva, direction, train_id, and timestamp.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the cleaned CSV file
    """
    print(f"Reading data from {input_file}...")
    
    try:
        # Read the raw data
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} records")
        
        # Make a copy to avoid warnings
        clean_df = df.copy()
        
        # Remove duplicates based on station_eva, direction, train_id, and timestamp
        # These fields should uniquely identify a train arrival/departure event
        print("\nRemoving duplicate records...")
        original_count = len(clean_df)
        clean_df = clean_df.drop_duplicates(subset=['station_eva', 'direction', 'train_id', 'timestamp'])
        duplicate_count = original_count - len(clean_df)
        print(f"Removed {duplicate_count} duplicate records. {len(clean_df)} records remain.")
        
        # Debug: Display the first few rows of raw data
        print("\nFirst few rows of raw data:")
        print(clean_df[['timestamp', 'tts']].head())
        
        # Process timestamps
        print("\nProcessing timestamps...")
        
        # Handle the timestamp field (planned time)
        # Format appears to be YYMMDDHHMM - e.g., 2504300820 means 2025-04-30 08:20
        def parse_timestamp(ts):
            if pd.isna(ts) or not ts:
                return pd.NaT
            
            ts_str = str(ts)
            # Debug individual timestamp parsing
            print(f"Parsing timestamp: {ts_str}")
            
            if len(ts_str) < 10:  # Ensure we have enough digits
                print(f"  - Too short, skipping")
                return pd.NaT
                
            try:
                year = int("20" + ts_str[0:2])  # Add "20" prefix for year
                month = int(ts_str[2:4])
                day = int(ts_str[4:6])
                hour = int(ts_str[6:8])
                minute = int(ts_str[8:10])
                
                parsed_date = pd.Timestamp(year, month, day, hour, minute)
                print(f"  - Parsed as: {parsed_date}")
                return parsed_date
            except Exception as e:
                print(f"  - Error: {str(e)}")
                return pd.NaT
        
        # Process the tts field which contains the actual time
        # Format appears to be YY-MM-DD HH:MM:SS.mmm - e.g., "25-04-30 08:20:34.939"
        def parse_tts(tts):
            if pd.isna(tts) or not tts:
                return pd.NaT
                
            tts_str = str(tts)
            # Debug individual tts parsing
            print(f"Parsing tts: {tts_str}")
            
            try:
                # Format: YY-MM-DD HH:MM:SS.mmm
                match = re.match(r'^(\d{2})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\.(\d{3})$', tts_str)
                
                if match:
                    year, month, day, hour, minute, second, ms = match.groups()
                    year = int("20" + year)  # Add "20" prefix for year
                    parsed_date = pd.Timestamp(year, int(month), int(day), int(hour), int(minute), int(second), int(ms) * 1000)
                    print(f"  - Parsed as: {parsed_date}")
                    return parsed_date
                else:
                    # Try with pandas parser as fallback
                    parsed_date = pd.to_datetime(tts_str, format="%y-%m-%d %H:%M:%S.%f", errors='coerce')
                    print(f"  - Parsed with fallback as: {parsed_date}")
                    return parsed_date
            except Exception as e:
                print(f"  - Error: {str(e)}")
                return pd.NaT
        
        # Take a small sample to debug the parsing
        sample_df = clean_df.head(2)
        
        # Apply the parsers to the sample
        print("\nParsing timestamps for a sample:")
        sample_df['planned_datetime'] = sample_df['timestamp'].apply(parse_timestamp)
        sample_df['actual_datetime'] = sample_df['tts'].apply(parse_tts)
        
        # Show the parsed sample results
        print("\nSample parsing results:")
        print(sample_df[['timestamp', 'planned_datetime', 'tts', 'actual_datetime']])
        
        # Now process the full dataset with fixed parsers
        print("\nNow processing the full dataset...")
        
        # Based on the debug output, update the parsers if needed
        def parse_timestamp_final(ts):
            if pd.isna(ts) or not ts:
                return pd.NaT
            
            ts_str = str(ts)
            if len(ts_str) < 10:
                return pd.NaT
                
            try:
                year = int("20" + ts_str[0:2])
                month = int(ts_str[2:4])
                day = int(ts_str[4:6])
                hour = int(ts_str[6:8])
                minute = int(ts_str[8:10])
                
                return pd.Timestamp(year, month, day, hour, minute)
            except:
                return pd.NaT
        
        def parse_tts_final(tts):
            if pd.isna(tts) or not tts:
                return pd.NaT
                
            tts_str = str(tts)
            try:
                match = re.match(r'^(\d{2})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\.(\d{3})$', tts_str)
                
                if match:
                    year, month, day, hour, minute, second, ms = match.groups()
                    year = int("20" + year)
                    return pd.Timestamp(year, int(month), int(day), int(hour), int(minute), int(second), int(ms) * 1000)
                else:
                    return pd.to_datetime(tts_str, format="%y-%m-%d %H:%M:%S.%f", errors='coerce')
            except:
                return pd.NaT
        
        # Apply the final parsers
        clean_df['planned_datetime'] = clean_df['timestamp'].apply(parse_timestamp_final)
        clean_df['actual_datetime'] = clean_df['tts'].apply(parse_tts_final)
        
        # Check for any unexpected dates
        planned_dates = clean_df['planned_datetime'].dt.date.unique()
        print("\nUnique planned dates found:")
        for date in sorted(planned_dates):
            print(f"  - {date}")
        
        # Removed date range filter - we're keeping all dates now
        
        print(f"After processing, {len(clean_df)} records remain")
        
        # Calculate delay in minutes
        print("Calculating delays...")
        clean_df['delay_seconds'] = (clean_df['actual_datetime'] - clean_df['planned_datetime']).dt.total_seconds()
        clean_df['delay_minutes'] = clean_df['delay_seconds'] / 60
        
        # Round delay to 1 decimal place
        clean_df['delay_minutes'] = clean_df['delay_minutes'].round(1)
        
        # Extract train type and number
        print("Extracting train information...")
        
        def extract_train_type(train_id):
            if pd.isna(train_id) or not train_id:
                return ""
            # Typically formats like: r17015112, IC123, etc.
            match = re.match(r'^([a-zA-Z]+)(\d+)$', str(train_id))
            if match:
                return match.group(1).upper()
            return ""
            
        def extract_train_number(train_id):
            if pd.isna(train_id) or not train_id:
                return ""
            # Extract the numeric part
            match = re.match(r'^[a-zA-Z]+(\d+)$', str(train_id))
            if match:
                return match.group(1)
            return ""
        
        clean_df['train_type'] = clean_df['train_id'].apply(extract_train_type)
        clean_df['train_number'] = clean_df['train_id'].apply(extract_train_number)
        
        # Add status field based on delay
        def get_status(delay):
            if pd.isna(delay):
                return 'Unknown'
            if delay < 1:
                return 'On time'
            elif delay < 5:
                return 'Slight delay'
            elif delay <= 15:
                return 'Moderate delay'
            else:
                return 'Significant delay'
                
        clean_df['status'] = clean_df['delay_minutes'].apply(get_status)
        
        # Add readable date and time fields
        clean_df['planned_date'] = clean_df['planned_datetime'].dt.strftime('%Y-%m-%d')
        clean_df['planned_time'] = clean_df['planned_datetime'].dt.strftime('%H:%M')
        clean_df['actual_date'] = clean_df['actual_datetime'].dt.strftime('%Y-%m-%d')
        clean_df['actual_time'] = clean_df['actual_datetime'].dt.strftime('%H:%M')
        
        # Add day of week
        clean_df['day_of_week'] = clean_df['planned_datetime'].dt.day_name()
        
        # Clean up and select only useful columns
        final_columns = [
            'station_eva', 
            'direction', 
            'train_type', 
            'train_number',
            'planned_date', 
            'planned_time', 
            'actual_date', 
            'actual_time',
            'delay_minutes', 
            'status', 
            'day_of_week'
        ]
        
        final_df = clean_df[final_columns].copy()
        
        # Sort the data chronologically
        print("Sorting data chronologically...")
        final_df = final_df.sort_values(by=['planned_date', 'planned_time'])
        
        # Double check the dates one more time
        print("\nDates in final dataset:")
        for date in sorted(final_df['planned_date'].unique()):
            print(f"  - {date}")
        
        # Save the cleaned data
        print(f"Saving cleaned data to {output_file}...")
        final_df.to_csv(output_file, index=False)
        
        # Print statistics
        print("\n--- Statistics ---")
        print(f"Total records: {len(final_df)}")
        
        # Count by direction
        direction_counts = final_df['direction'].value_counts()
        print(f"\nDirection counts:")
        for direction, count in direction_counts.items():
            print(f"  {direction}: {count}")
        
        # Delay statistics
        print(f"\nDelay statistics:")
        delay_stats = final_df['delay_minutes'].describe()
        print(f"  Average delay: {delay_stats['mean']:.1f} minutes")
        print(f"  Median delay: {delay_stats['50%']:.1f} minutes")
        print(f"  Max delay: {delay_stats['max']:.1f} minutes")
        
        # Status distribution
        status_counts = final_df['status'].value_counts()
        print(f"\nStatus distribution:")
        for status, count in status_counts.items():
            percentage = (count / len(final_df)) * 100
            print(f"  {status}: {count} ({percentage:.1f}%)")
        
        # Print date summary
        dates = final_df['planned_date'].unique()
        dates.sort()  # Make sure they're in order
        print(f"\nDates in dataset ({len(dates)}):")
        for date in dates:
            day_data = final_df[final_df['planned_date'] == date]
            day_of_week = day_data['day_of_week'].iloc[0]
            count = len(day_data)
            avg_delay = day_data['delay_minutes'].mean().round(1)
            print(f"  {date} ({day_of_week}): {count} trains, avg delay: {avg_delay} min")
            
        print(f"\nCleaning completed successfully. Data saved to {output_file}")
        
        return final_df
        
    except Exception as e:
        print(f"Error cleaning data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the cleaning function
    clean_df = clean_train_data()
    
    # Display sample of the cleaned data
    if clean_df is not None:
        print("\n--- Sample of Cleaned Data ---")
        print(clean_df.head())