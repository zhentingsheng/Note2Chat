import json
import re
import demjson3

class TextExtractor:

    def extract_first_json_object(self, text):
        start_index = text.find('{')
        if start_index == -1:
            return None

        brace_count = 0
        end_index = -1

        for i in range(start_index, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_index = i
                    break

        if end_index == -1:
            return None

        json_string = text[start_index:end_index + 1].strip()

        try:
            json_data = demjson3.decode(json_string)
            return json_data
        except demjson3.JSONDecodeError:
            return None
        

    def extract_first_json_array(self, text):
        start_index = text.find('[')
        if start_index == -1:
            return None

        bracket_count = 0
        end_index = -1

        for i in range(start_index, len(text)):
            if text[i] == '[':
                bracket_count += 1
            elif text[i] == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    end_index = i
                    break

        if end_index == -1:
            return None

        json_string = text[start_index:end_index + 1].strip()

        try:
            json_data = demjson3.decode(json_string)
            return json_data
        except demjson3.JSONDecodeError:
            return None
