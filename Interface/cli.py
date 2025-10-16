import os
import random
import time
import pygame

HIGH_SCORE_FILE = "high_score.txt"


def play_audio(file):
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    pygame.mixer.music.load(file)  # compressed WAV works
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

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


def play_game(highScore):
    currentScore = 0
    print(f"Current score: {currentScore}")

    while True:
        expectedClass = random.randint(0, 15)

        # Play audio file
        audioFile = f"./../Interface/audio/{expectedClass}.wav"
        if os.path.exists(audioFile):
            play_audio(audioFile)
        else:
            print(f"(Audio file {audioFile} missing — skipping sound)")

        time.sleep(2)  # wait 2 seconds after audio

        try:
            predictedClass = int(input("Predicted class from model: ").strip())
        except ValueError:
            print("Invalid input! Please enter a number.")
            continue

        if predictedClass == expectedClass:
            currentScore += 1
            print(f"Correct! Current score: {currentScore}")
        else:
            print("\nGame over!")
            print(f"Final score: {currentScore}")
            if currentScore > highScore:
                highScore = currentScore
                save_high_score(highScore)
            print(f"High score: {highScore}")
            break

    return highScore


def main():
    highScore = load_high_score()
    print(f"High score: {highScore}")
    
    while True:
        input("Press Enter to start game: ")
        highScore = play_game(highScore)


if __name__ == "__main__":
    main()
