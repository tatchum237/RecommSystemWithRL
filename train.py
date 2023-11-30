from random import randrange
from model import LinUCB
from utils import load_data, preprocess_fast, preprocess_without_title, preprocess_with_title, get_next_song_context

def main():
    # Load and preprocess data
    print("preprocessing data...")
    data = load_data("data/songs.csv")
    features = preprocess_fast(data)
    

    # Initialize your LinUCB model
    num_actions = len(features)
    num_features = features.shape[1]
    alpha = 0.9  # You can adjust this value based on your needs
    linucb_model = LinUCB(num_actions=num_actions, num_features=num_features, alpha=alpha)
    chosen_action = randrange(num_actions)

    # Online training loop
    while True:
        title, context = get_next_song_context(data, features, chosen_action)
        
        # if context is None:
        #     # Exit the loop when there are no more songs
        #     break

        chosen_action = linucb_model.choose_action(context)

        # Simulate user providing feedback for the chosen action (song)
        print(f"Please rate the song titled '{title}' on a scale from 1 to 10:")
        user_rating = int(input())
        
        # Update the model with the user feedback
        linucb_model.update_model(context=context, action=chosen_action, reward=user_rating)

    # Save the trained model for later use (optional)
    # linucb_model.save_model("trained_model.pkl")

if __name__ == "__main__":
    main()
