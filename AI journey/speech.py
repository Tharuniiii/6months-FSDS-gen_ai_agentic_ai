import speech_recognition as sr
import pyttsx3 as pt
import pywhatkit as pk

listening = sr.Recognizer()
engine = pt.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def hear():
    cmd = ""  # Initialize cmd so it always exists
    try:
        with sr.Microphone() as mic:
            print('listening....')
            voice = listening.listen(mic)
            cmd = listening.recognize_google(voice)
            cmd = cmd.lower()
            if 'kodi' in cmd:
                cmd = cmd.replace('kodi', '')
                print(f"Command after removing wake word: {cmd}")
    except Exception as e:
        print(f"Error: {e}")
    return cmd

def run():
    cmd = hear()
    print(f"Received command: {cmd}")
    if 'play' in cmd:
        song = cmd.replace('play', '').strip()
        speak('playing ' + song)
        pk.playonyt(song)
    else:
        speak("Sorry, I didn't understand the command.")

if __name__ == "__main__":
    run()