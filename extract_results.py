import re
#           0              1               2         3                 4                     5                6               7                  8                        9
model = ['Offline','Online_Finetuning','Edison','G-Baseline','G-Baseline_Contrastive','G-Baseline_NCE','Edison_DeepLSTM', 'Offline_NCE', 'Online_Finetuning_NCE', 'Baseline_NCE_WFR']
dataset = ['wisdm','realworld','mhealth','pamap']
architecture = ['tinyhar','resnet18','DeepLSTM']

model_idxs = [0,1,3,4,5, 7, 8, 9] #[0,1,3,4,5] #[0,1,3,4,5]
dataset_idx = 1
architecture_idx = 1
print("Dataset:",dataset[dataset_idx])
print("Architecture",architecture[architecture_idx])
for model_idx in model_idxs:
    
    data = open(f"/netscratch/dmittal/continual_learning/Saved_Models/Incremental/{model[model_idx]}/{dataset[dataset_idx]}/results_{dataset[dataset_idx]}_{architecture[architecture_idx]}.log", "r")
    
    data = data.read()
    
    
    # Regex to extract the results summary
    pattern = re.compile(
        r"Initial Stage ALL Acc Mean: ([\d.]+) Std: ([\d.]+) F1 Mean: ([\d.]+) Std: ([\d.]+)\n"
        r"Before CL Seen Acc Mean: ([\d.]+) Std: ([\d.]+) F1 Mean: ([\d.]+) Std: ([\d.]+)\n"
        r"After CL Seen Acc Mean: ([\d.]+) Std: ([\d.]+) F1 Mean: ([\d.]+) Std: ([\d.]+)\n"
        r"After CL Unseen Acc Mean: ([\d.]+) Std: ([\d.]+) F1 Mean: ([\d.]+) Std: ([\d.]+)\n"
        r"After CL Overall Acc Mean: ([\d.]+) Std: ([\d.]+) F1 Mean: ([\d.]+) Std: ([\d.]+)"
    )
    n_values = 20
# Extract the data
    matches = pattern.search(data)

    
    print("\nModel:", model[model_idx])
    
    
    if matches:
        results_array = [float(matches.group(i)) for i in range(1, n_values+1)]
        
        formatted_results_array = [f"{num:.4f}" for num in results_array]
        # Convert the array to a comma-separated string
        results_string = ', '.join(map(str, formatted_results_array))
        
        # Print the results string (this can be copied and pasted into an Excel row)
        # if(model_idx==0):
        #     zeros_to_insert = [0.0, 0.0, 0.0, 0.0]
        # # Insert the zeros after the 4th index
        #     modified_list = formatted_results_array[:4] + zeros_to_insert + formatted_results_array[4:]
        #     results_string = ', '.join(f"{float(num):.4f}" for num in modified_list)
        print(results_string)
    else:
        print("No match found.")