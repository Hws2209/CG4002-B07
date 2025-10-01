import os
import random
import time
import winsound

HIGH_SCORE_FILE = "high_score.txt"


def play_audio(file):
    winsound.PlaySound(file, winsound.SND_FILENAME)

def load_high_score():
    if not os.path.exists(HIGH_SCORE_FILE):
        with open(HIGH_SCORE_FILE, "w") as f:
            f.write("0")
        return 0
    with open(HIGH_SCORE_FILE, "r") as f:
        return int(f.read().strip() or 0)


def save_high_score(score):
    with open(HIGH_SCORE_FILE, "w") as f:
        f.write(str(score))


def play_game(high_score):
    current_score = 0
    print(f"Current score: {current_score}")

    while True:
        expected_class = random.randint(0, 15)

        # Play audio file
        audio_file = f"audio/{expected_class}.wav"
        if os.path.exists(audio_file):
            play_audio(audio_file)
        else:
            print(f"(Audio file {audio_file} missing — skipping sound)")

        time.sleep(2)  # wait 2 seconds after audio

        try:
            predicted_class = int(input("Predicted class from model: ").strip())
        except ValueError:
            print("Invalid input! Please enter a number.")
            continue

        if predicted_class == expected_class:
            current_score += 1
            print(f"Correct! Current score: {current_score}")
        else:
            print("\nGame over!")
            print(f"Final score: {current_score}")
            if current_score > high_score:
                high_score = current_score
                save_high_score(high_score)
            print(f"High score: {high_score}")
            break

    return high_score


def main():
    high_score = load_high_score()
    print(f"High score: {high_score}")
    
    while True:
        input("Press Enter to start game: ")
        high_score = play_game(high_score)


if __name__ == "__main__":
    main()
