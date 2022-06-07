from gui import WindowClient
from sentiment_analyzer import SentimentAnalyzer


"""
EXAMPLES:

[Negative]:
This might be the poorest example of amateur propaganda ever made. The writers and producers should study the German films of the thirties and forties. They knew how to sell. Even soviet-style clunky leader as god-like father-figure were better done. Disappointing. The loss of faith, regained in church at last second just in time for daddy to be "saved" by the Hoover/God was not too bad. Unfortunately, it seemed rushed and not nearly melodramatic enough. A few misty heavenlier shots of the angelical Hoover up in the corner of the screen-beaming and nodding- would have added a lot. The best aspect is Hoover only saving the deserving family and children WHO had "proven" their worth. Unfortunately, other poor homeless were portrayed as likable and even good- yet the Hoover-God doesn\'t help them. A better approach would have been shots of them drinking spirits to show the justice of their condition. Finally, bright and cheerful scenes of recovery (after Hoover saved the country from the depression) should have rolled at the end. We could see then how Hoover-God had saved not just THIS deserving family, but all the truly deserving. Amateurist at best.

[Positive]:
Hilarious, evocative, confusing, brilliant film. Reminds me of Bunuel's L'Age D'Or or Jodorowsky's Holy Mountain-- lots of strange characters mucking about and looking for..... what is it? I laughed almost the whole way through, all the while keeping a peripheral eye on the bewildered and occasionally horrified reactions of the audience that surrounded me in the theatre. Entertaining through and through, from the beginning to the guts and poisoned entrails all the way to the end, if it was an end. I only wish i could remember every detail. It haunts me sometimes.<br /><br />Honestly, though, i have only the most positive recollections of this film. As it doesn\'t seem to be available to take home and watch, i suppose i'll have to wait a few more years until Crispin Glover comes my way again with his Big Slide Show (and subsequent "What is it?" screening)... I saw this film in Atlanta almost directly after being involved in a rather devastating car crash, so i was slightly dazed at the time, which was perhaps a very good state of mind to watch the prophetic talking arthropods and the retards in the superhero costumes and godlike Glover in his appropriate burly-Q setting, scantily clad girlies rising out of the floor like a magnificent DADAist wet dream.<br /><br />Is it a statement on Life As We Know It? Of course everyone EXPECTS art to be just that. I rather think that the truth is more evident in the absences and in the negative space. What you don\'t tell us is what we must deduce, but is far more valid than the lies that other people feed us day in and day out. Rather one "WHAT IS IT?" than 5000 movies like "Titanic" or "Sleepless in Seattle" (shudder, gag, groan).<br /><br />Thank you, Mr. Glover (additionally a fun man to watch on screen or at his Big Slide Show-- smart, funny, quirky, and outrageously hot). Make more films, write more books, keep the nightmare alive.
"""

MODEL_PATH = '../model/model1'
analyzer = SentimentAnalyzer(model_path=MODEL_PATH)
window = WindowClient(engine=analyzer)
window.launch()
