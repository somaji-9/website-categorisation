import pandas as pd

def balance_csv(input_file, output_balanced, output_extra, margin=83):
    # Load your dataset
    df = pd.read_csv(input_file)
    
    balanced_list = []
    extra_list = []
    
    # Group by category to handle each group individually
    grouped = df.groupby('category')
    
    for category, group in grouped:
        if len(group) > margin:
            # Keep the first 83 rows
            balanced_list.append(group.iloc[:margin])
            # Move the rest to the extra list
            extra_list.append(group.iloc[margin:])
        else:
            # If less than or equal to margin, keep the whole group
            balanced_list.append(group)
    
    # Combine the lists back into DataFrames
    balanced_df = pd.concat(balanced_list, ignore_index=True)
    
    # Check if there are any extra rows to save
    if extra_list:
        extra_df = pd.concat(extra_list, ignore_index=True)
        extra_df.to_csv(output_extra, index=False)
        print(f"Saved {len(extra_df)} extra rows to {output_extra}")
    else:
        print("No categories exceeded the margin.")
        
    # Save the balanced file
    balanced_df.to_csv(output_balanced, index=False)
    print(f"Saved balanced dataset to {output_balanced}")

# Run the function
balance_csv('raw_data.csv', 'balanced_data.csv', 'extra_rows.csv')