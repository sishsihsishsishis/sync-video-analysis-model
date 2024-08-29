import logging
import requests
from retrying import retry
from concurrent.futures import ThreadPoolExecutor
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GptServiceImpl:
    MAX_RETRIES = 3
    API_KEY = "sk-proj-sK7ff3WWXS98BPfyYEp3T3BlbkFJPQ1oNsNgdFAxFEY9uFGs"
    URL = "https://api.openai.com/v1/chat/completions"

    PROMPT_PRE = """
    You will be provided with a conversation formatted as:  
    <speaker>\\t<starts>\\t<ends>\\t<sentence>

    Task:  
    1. Divide the conversation into 1 or 2 chapters based on themes or ideas.  
    2. Summarize each chapter in one sentence and provide 3 bullet points for each.  
    3. Include the actual start and end times for each chapter from the conversation.

    **Important:**  
    - **Use `**` for bullet points.** Ensure each bullet point starts with `**` followed by a space.
    - **Do not use placeholders for start and end times.** Use the actual start and end times from the input.  
    - **Do not use placeholders for summaries and bullet points.** Provide actual content derived from the input.

    **Output Format:**

    Chapter 1: [Actual Summary Here] (Actual Start Time - Actual End Time)  
    ** Bullet Point 1  
    ** Bullet Point 2  
    ** Bullet Point 3  
    ||||  
    Chapter 2: [Actual Summary Here] (Actual Start Time - Actual End Time)  
    ** Bullet Point 1  
    ** Bullet Point 2  
    ** Bullet Point 3  

    **Example Input:**  
    Speaker1\\t00:00\\t00:10\\tHello, how are you?  
    Speaker2\\t00:11\\t00:20\\tI’m good, thank you! How about you?  
    Speaker1\\t00:21\\t00:30\\tI’m doing well, just busy with work.  
    Speaker2\\t00:31\\t00:40\\tI understand, it’s been a hectic week.  
    Speaker1\\t00:41\\t00:50\\tAbsolutely, but we’ll manage.

    **Example Output:**  

    Chapter 1: Introductions and Greetings (00:00 - 00:20)  
    ** The speakers greet each other.  
    ** They ask about each other's well-being.  
    ** The conversation starts positively.  
    ||||  
    Chapter 2: Discussing Work (00:21 - 00:50)  
    ** Speaker 1 talks about being busy with work.  
    ** Speaker 2 agrees and mentions the hectic schedule.  
    ** They acknowledge the challenges but remain positive.

    **Do not use placeholders or example text in your output. Ensure that you provide actual summaries, start and end times, and bullet points based on the provided conversation data.**
    """

    PROMPT_USER_HIGHLIGHT = """
    # role #
    You, as a highly skilled AI expert trained in language comprehension and summarization, are able to fully read and understand and accurately analyze long segments of conversational text.

    # objective #
    After fully understanding and comprehending the dialog text, select three highlights of about 30 seconds each for each speaker from all the information, not just the first half, and if a speaker doesn't have a highlight, select the better 30 seconds of what he said and output it, outputting the speaker, the start time, and the end time, and providing a short Description.

    # style #
    Must be accurate and robust, no matter how many times you output this type of result.

    # tone #
    Accurate and persuasive

    # audience #
    People who are interested in the team's online communication

    # response #
    Containing only the SPEAKER number, start time, end time, and a description of the segment, to be output for all attendees, the specific template for one of the attendees is as follows:
    speaker01
    start time: 25.23
    end time: 55.16
    Description: This clip focuses on the current hotness of the product in the market and emphasizes the future development strategy.

    speaker01
    start time：598.78
    end time: 628.16
    Description: This clip describes the recent problems in the personnel department and proposes solutions to them.

    speaker01
    start time：1945.45
    end time:1975.26
    Description: This clip summarizes the meeting and presents the next steps.

    speaker02
    start time: 25.23
    end time: 55.16
    Description: This clip focuses on the current hotness of the product in the market and emphasizes the future development strategy.

    speaker02
    start time：598.78
    end time: 628.16
    Description: This clip describes the recent problems in the personnel department and proposes solutions to them.

    speaker02
    start time：1945.45
    end time:1975.26
    Description: This clip summarizes the meeting and presents the next steps.
    """

    def __init__(self, api_key=None, url=None):
        if api_key:
            self.API_KEY = api_key
        if url:
            self.URL = url

    @retry(stop_max_attempt_number=MAX_RETRIES, wait_fixed=1000)
    def summary_nlp(self, nlp_data, meeting_id, delete_map):
        nlp_summary_list = []
        contents = []
        for count, nlp_datum in enumerate(nlp_data):
            contents.append("\t".join(nlp_datum))
            if (count + 1) % 300 == 0:
                self.process_contents(contents, nlp_summary_list, meeting_id)
                contents.clear()
        # print("contents", contents)
        if contents:
            self.process_contents(contents, nlp_summary_list, meeting_id)
        return nlp_summary_list

    def process_contents(self, contents, nlp_summary_list, meeting_id):
        final_message = "\n".join(contents)
        logger.info(final_message)
        response = self.send_message_to_gpt(final_message, self.PROMPT_PRE)
        self.parse_nlp_summary(response, meeting_id, nlp_summary_list)

    def send_message_to_gpt(self, content, system_prompt):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.API_KEY}"
        }
        data = {
            "model": "gpt-3.5-turbo-16k",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ]
        }
        response = requests.post(self.URL, json=data, headers=headers)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

    def parse_nlp_summary(self, input_text, meeting_id, nlp_summary_list):
        # Split the input text by chapter delimiter
        chapters = input_text.split("||||")
        logging.info(input_text)
        for chapter in chapters:
            chapter = chapter.strip()
            if not chapter:
                continue
            
            # Split the chapter into lines
            lines = chapter.split("\n")
            
            # Extract the chapter summary and time range
            summary_line = lines[0].strip()
            
            # Extract the summary, start, and end times
            summary_parts = summary_line.split(" (")
            summary = summary_parts[0].strip()
            time_range = summary_parts[1][:-1]  # Remove the trailing ')'
            start_time, end_time = time_range.split(" - ")
            
            # Extract bullet points
            bullets = [line.strip() for line in lines[1:] if line.strip().startswith('** ')]
            bullets = [bullet[2:].strip() for bullet in bullets]  # Remove '- ' from each bullet point
            
            # Create the summary dictionary
            nlp_summary = {
                "summary": summary,
                "start": start_time,
                "end": end_time,
                "bullets": bullets,
            }
            
            nlp_summary_list.append(nlp_summary)

    @retry(stop_max_attempt_number=MAX_RETRIES, wait_fixed=1000)
    def summary_nlp_text(self, nlp_data, nlp_tables, meeting_id, delete_map):
        try:
            content = "\n".join("\t".join(nlp_datum) for nlp_datum in nlp_data)
            final_message = content
            logger.info(final_message)
            response = self.send_message_to_gpt(final_message, self.PROMPT_PRE)
            summaries = response.split("\n\n")
        except Exception as e:
            logger.error(f"Error in summary_nlp_text: {str(e)}")
            raise

    def process_team_highlight(self, nlp_data, meeting_id):
        logger.info(f"[GptServiceImpl][processTeamHighlight] meetingID :{meeting_id}")
        data = self.concatenate_nlp_data(nlp_data, 120)
        result = [self.send_message_to_gpt(datum, self.PROMPT_TEAM_HIGHLIGHT) for datum in data]
        section_list = self.parse_3_team_highlight(result, meeting_id)

    def process_user_highlight(self, nlp_data, meeting_id):
        logger.info(f"[GptServiceImpl][processUserHighlight] meetingID :{meeting_id}")
        data = self.concatenate_nlp_data(nlp_data, 120)
        result = [self.send_message_to_gpt(datum, self.PROMPT_USER_HIGHLIGHT) for datum in data]
        section_list = self.parse_3_user_highlight(result, meeting_id)

    def parse_3_user_highlight(self, result, meeting_id):
        section_list = []
        map_ = {}
        for highlight in result:
            split = highlight.split("\n\n")
            for content in split:
                lines = content.splitlines()
                if len(lines) == 4 and all(":" in line for line in lines[1:4]):
                    speaker = lines[0]
                    start_time = float(lines[1].split(":")[1].strip())
                    end_time = float(lines[2].split(":")[1].strip())
                    description = lines[3].split(":")[1].strip()
                    section = {"meeting_id": meeting_id, "start_time": start_time, "end_time": end_time, "speaker": speaker, "description": description}
                    map_.setdefault(speaker, []).append(section)
        for sections in map_.values():
            section_list.extend(sections[:3])
        return section_list

    def parse_3_team_highlight(self, result, meeting_id):
        section_list = []
        for highlight in result:
            split = highlight.split("\n\n")
            for content in split:
                lines = content.splitlines()
                if len(lines) == 3 and all(":" in line for line in lines):
                    start_time = float(lines[0].split(":")[1].strip())
                    end_time = float(lines[1].split(":")[1].strip())
                    description = lines[2].split(":")[1].strip()
                    section = {"meeting_id": meeting_id, "start_time": start_time, "end_time": end_time, "speaker": "team", "description": description}
                    section_list.append(section)
        return section_list[:3]

    def concatenate_nlp_data(self, nlp_data, n):
        concatenated_data = []
        sb = []
        count = 0
        for data_array in nlp_data:
            sb.append("\t".join(data_array[:4]))
            count += 1
            if count % n == 0:
                concatenated_data.append("\n".join(sb))
                sb.clear()
        if sb:
            concatenated_data.append("\n".join(sb))
        return concatenated_data

# if __name__ == "__main__":
#     nlp_data = [
#             {
#                 "speaker": "speaker02",
#                 "starts": 0.0,
#                 "ends": 12.08,
#                 "sentence": " this is our weekly product marketing meeting and the first topic of conversation is one we've",
#                 "emotion": "neutral",
#                 "dialogue": "Statement-non-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 12.08,
#                 "ends": 20.5,
#                 "sentence": " chatted about a few times and you know probably bears worth a little conversation. so i recorded",
#                 "emotion": "neutral",
#                 "dialogue": "Statement-non-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 20.5,
#                 "ends": 26.28,
#                 "sentence": " a video over the weekend. i posted the video and the deck and actually i will say this too. i should",
#                 "emotion": "neutral",
#                 "dialogue": "Statement-non-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 26.28,
#                 "ends": 33.14,
#                 "sentence": " have this as an agenda item and maybe even from last week. dara, if you haven't seen this, i think",
#                 "emotion": "neutral",
#                 "dialogue": "Statement-non-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 33.14,
#                 "ends": 39.46,
#                 "sentence": " dara is putting together a training. actually it's a larger motion. so the kind of idea is in order to",
#                 "emotion": "neutral",
#                 "dialogue": "Statement-non-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 39.46,
#                 "ends": 45.06,
#                 "sentence": " promote cross mobility within marketing, the idea is that there would be some kind of training",
#                 "emotion": "neutral",
#                 "dialogue": "Statement-non-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 45.06,
#                 "ends": 51.900000000000006,
#                 "sentence": " curriculum and if you wanted to shift like laterally into a different role, the idea is that",
#                 "emotion": "neutral",
#                 "dialogue": "Statement-non-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 51.900000000000006,
#                 "ends": 54.260000000000005,
#                 "sentence": " you should be able to complete all of this stuff.",
#                 "emotion": "neutral",
#                 "dialogue": "Statement-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 55.3,
#                 "ends": 56.260000000000005,
#                 "sentence": " now i like this.",
#                 "emotion": "positive",
#                 "dialogue": "Statement-non-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 56.28,
#                 "ends": 62.1,
#                 "sentence": " i like this a lot. strategic marketing in the time i've been at gitlab has not hired anyone else from",
#                 "emotion": "positive",
#                 "dialogue": "Statement-non-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 62.1,
#                 "ends": 67.82000000000001,
#                 "sentence": " any other part of the company and i think that's a huge tragedy. so i would love to see folks from",
#                 "emotion": "neutral",
#                 "dialogue": "Statement-non-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 67.82000000000001,
#                 "ends": 72.38,
#                 "sentence": " other parts of marketing and other parts of the org move into strategic marketing or our team in",
#                 "emotion": "neutral",
#                 "dialogue": "Statement-non-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 72.38,
#                 "ends": 79.36,
#                 "sentence": " particular. and probably i also think of product marketers, part of our role is to know a little",
#                 "emotion": "neutral",
#                 "dialogue": "Statement-non-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 79.36,
#                 "ends": 84.22,
#                 "sentence": " bit about everything. and so as other leaders are kind of putting together like demandgen 101",
#                 "emotion": "neutral",
#                 "dialogue": "Abandoned or Turn-Exit"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 84.22,
#                 "ends": 86.26,
#                 "sentence": " or like sdrs 101, i think that's a huge tragedy. i think that's a huge tragedy. i think that's a huge",
#                 "emotion": "negative",
#                 "dialogue": "Statement-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 86.28,
#                 "ends": 87.28,
#                 "sentence": " challenge for us to be able to do this kind of thing. so i think that's a huge challenge for us.",
#                 "emotion": "negative",
#                 "dialogue": "Abandoned or Turn-Exit"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 87.28,
#                 "ends": 88.28,
#                 "sentence": " and i think that's the key to us is we have to do this, but it's really key. and so i think you can",
#                 "emotion": "neutral",
#                 "dialogue": "Statement-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 88.28,
#                 "ends": 89.28,
#                 "sentence": " see that in the product marketing that we've been doing in the past. we've been doing it for the",
#                 "emotion": "neutral",
#                 "dialogue": "Abandoned or Turn-Exit"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 89.28,
#                 "ends": 90.28,
#                 "sentence": " last two years. so for us, this is the first time that we've done product marketing 101, like we",
#                 "emotion": "neutral",
#                 "dialogue": "Statement-non-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 90.28,
#                 "ends": 92.64,
#                 "sentence": " should consume all that content and just be familiar with it. and then we are also tasked with",
#                 "emotion": "neutral",
#                 "dialogue": "Statement-non-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 92.64,
#                 "ends": 98.9,
#                 "sentence": " creating a little bit of product marketing 101. so in that spirit, i linked a linkedin video.",
#                 "emotion": "neutral",
#                 "dialogue": "Abandoned or Turn-Exit"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 98.9,
#                 "ends": 103.68,
#                 "sentence": " did anybody watch any of that linkedin training that i linked last week and i've chatted with a few",
#                 "emotion": "neutral",
#                 "dialogue": "Yes-No-Question"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 103.68,
#                 "ends": 110.58,
#                 "sentence": " of you folks about it? did anybody check that out yet? no. totally go check this out. okay.",
#                 "emotion": "neutral",
#                 "dialogue": "Yes-No-Question"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 110.58,
#                 "ends": 113.68,
#                 "sentence": " so i was surprised at how good it was.",
#                 "emotion": "positive",
#                 "dialogue": "Statement-non-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 113.84,
#                 "ends": 114.56,
#                 "sentence": " i didn't expect it.",
#                 "emotion": "positive",
#                 "dialogue": "Statement-non-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 114.84,
#                 "ends": 119.72,
#                 "sentence": " i've had some really good linkedin learning content that i've accessed before at other",
#                 "emotion": "neutral",
#                 "dialogue": "Statement-non-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 119.72,
#                 "ends": 120.06,
#                 "sentence": " jobs.",
#                 "emotion": "neutral",
#                 "dialogue": "Abandoned or Turn-Exit"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 121.02,
#                 "ends": 123.54,
#                 "sentence": " and so i was like, oh, i'll just go see what do they have about product marketing.",
#                 "emotion": "neutral",
#                 "dialogue": "Statement-non-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 123.68,
#                 "ends": 127.96,
#                 "sentence": " and i found this one course and it's like b2b saas marketing 101.",
#                 "emotion": "neutral",
#                 "dialogue": "Statement-non-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 128.88,
#                 "ends": 134.06,
#                 "sentence": " and there's nothing in there that you haven't heard before, but it's just so good.",
#                 "emotion": "positive",
#                 "dialogue": "Statement-non-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 134.06,
#                 "ends": 136.72,
#                 "sentence": " it's all very compact.",
#                 "emotion": "neutral",
#                 "dialogue": "Statement-non-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 136.72,
#                 "ends": 141.44,
#                 "sentence": " i highly recommend if you just watch the course end to end, it's an hour.",
#                 "emotion": "positive",
#                 "dialogue": "Statement-non-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 142.32,
#                 "ends": 145.1,
#                 "sentence": " 100% recommend just spending an hour and watching that course.",
#                 "emotion": "positive",
#                 "dialogue": "Abandoned or Turn-Exit"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 145.2,
#                 "ends": 149.02,
#                 "sentence": " it's a great just refresher and it'll share some things that you didn't think about before.",
#                 "emotion": "positive",
#                 "dialogue": "Statement-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 150.06,
#                 "ends": 155.18,
#                 "sentence": " and or if you want to take like a day or two and like do all of the homework or whatever",
#                 "emotion": "neutral",
#                 "dialogue": "Statement-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 155.18,
#                 "ends": 158.32,
#                 "sentence": " and really dig into it, highly recommended that.",
#                 "emotion": "neutral",
#                 "dialogue": "Abandoned or Turn-Exit"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 158.7,
#                 "ends": 162.28,
#                 "sentence": " so in that vein, that course will go into like the, if you want to be a product marketer,",
#                 "emotion": "neutral",
#                 "dialogue": "Statement-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 162.34,
#                 "ends": 163.66,
#                 "sentence": " this is stuff you should know about product marketing.",
#                 "emotion": "neutral",
#                 "dialogue": "Statement-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 164.02,
#                 "ends": 166.62,
#                 "sentence": " i also recorded this video on what is the",
#                 "emotion": "neutral",
#                 "dialogue": "Statement-non-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 166.62,
#                 "ends": 166.7,
#                 "sentence": " difference between a product marketing course and a saas marketing course.",
#                 "emotion": "neutral",
#                 "dialogue": "Statement-non-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 166.7,
#                 "ends": 169.17999999999998,
#                 "sentence": " so it's a difference between a product service and solution.",
#                 "emotion": "neutral",
#                 "dialogue": "Statement-opinion"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 170.1,
#                 "ends": 178.26,
#                 "sentence": " on that vein, cindy, i think you maybe also had an mr about gt emotions or solutions.",
#                 "emotion": "neutral",
#                 "dialogue": "Abandoned or Turn-Exit"
#             },
#             {
#                 "speaker": "speaker02",
#                 "starts": 178.32,
#                 "ends": 179.29999999999998,
#                 "sentence": " do you want to share your thoughts there?",
#                 "emotion": "neutral",
#                 "dialogue": "Yes-No-Question"
#             },
#             {
#                 "speaker": "speaker00",
#                 "starts": 179.95999999999998,
#                 "ends": 181.02,
#                 "sentence": " yeah, i just linked it.",
#                 "emotion": "neutral",
#                 "dialogue": "Statement-non-opinion"
#             },
#             {
#                 "speaker": "speaker00",
#                 "starts": 182.52,
#                 "ends": 188.88,
#                 "sentence": " so we've got what i was looking at is the what we call a solution.",
#                 "emotion": "neutral",
#                 "dialogue": "Statement-opinion"
#             },
#     ]
#     meeting_id = "12345"
#     delete_map = {}  # Replace with your actual delete map

#     # Instantiate the GptServiceImpl class
#     gpt_service = GptServiceImpl()

#     # Call the summary_nlp method
#     gpt_service.summary_nlp(nlp_data, meeting_id, delete_map)
