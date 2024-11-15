import json

# Define the file paths for your data
movie_lines_file = 'cornell_movie_data/cornell movie-dialogs corpus/movie_lines.txt'  # File containing the movie lines
conversations_file = 'cornell_movie_data/cornell movie-dialogs corpus/movie_conversations.txt'  # File containing the conversations
output_file = 'datasets/chatbot_dataset.json'  # Output file for the structured dataset

# Step 1: Read the movie lines into a dictionary and sort by line ID
lines_dict = {}
with open(movie_lines_file, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        parts = line.strip().split(' +++$+++ ')
        if len(parts) >= 5:  # Ensure there are enough parts
            line_id, user_id, movie_id, character, dialogue = parts
            lines_dict[line_id] = dialogue

# Sort the lines based on the line ID
sorted_lines = sorted(lines_dict.items(), key=lambda x: x[0])

# Create a new dictionary with sorted lines
sorted_lines_dict = {line_id: dialogue for line_id, dialogue in sorted_lines}

# Step 2: Read the conversations and generate source-response pairs
dataset = []
with open(conversations_file, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        parts = line.strip().split(' +++$+++ ')
        if len(parts) >= 4:  # Ensure there are enough parts
            user1_id, user2_id, movie_id, line_ids_str = parts
            line_ids = eval(line_ids_str)  # Convert string representation of list to actual list

            # Step 3: Create pairs from the line IDs
            for i in range(len(line_ids) - 1):
                source_line_id = line_ids[i]
                response_line_id = line_ids[i + 1]
                if source_line_id in sorted_lines_dict and response_line_id in sorted_lines_dict:
                    source = sorted_lines_dict[source_line_id]
                    response = sorted_lines_dict[response_line_id]

                    if len(source) < 190 and len(response) < 190:
                        # Append the source-response pair to the dataset
                        dataset.append({'source': source, 'response': response})

# Step 4: Write the dataset to a JSON file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print(f'Dataset successfully generated and saved to {output_file}.')