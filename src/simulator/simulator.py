import os
import json
from simulator.doctor_agent import DoctorAgent
from simulator.patient_agent import PatientAgent
from prompts.doctor import get_doctor_system_prompt
from prompts.patient import get_patient_system_prompt
from utils.extractor import TextExtractor
from utils.models import batch_call_gpt_server


class DoctorResponder:
    def generate_batch(self, conversations):
        raise NotImplementedError
    def close(self):
        print("doctor responser closed")
    
class GPTServerResponder(DoctorResponder):
    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name

    def generate_batch(self, conversations):
        return batch_call_gpt_server(
            convos=conversations,
            model=self.model_name,
            max_tokens=2048
        )
    
class LLMResponder(DoctorResponder):
    def __init__(self, doctor_agent, batch_size=1024, turn_mode="multi"):
        self.doctor_agent = doctor_agent
        self.batch_size = batch_size
        self.turn_mode = turn_mode

    def generate_batch(self, conversations):
        return self.doctor_agent.generate_responses_in_batches(
            conversations,
            batch_size=self.batch_size,
            turn_mode=self.turn_mode,
        )
    
    def close(self):
        self.doctor_agent.close()
    
class SimulatorBase:
    def __init__(
        self,
        doctor_responder: DoctorResponder,
        exp_name: str = '',
        max_turn: int = 20,
        conv_key: str = "conv_inference",
        system_instruct_version: str = "v1",
        turn_mode: str = "multi",
        patient_model: str = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
    ):
        self.doctor_responder = doctor_responder
        self.patient_agent = PatientAgent(patient_model)
        # self.patient_agent = PatientAgent()
        self.doctor_system_instruct = get_doctor_system_prompt(system_instruct_version)
        self.max_turn = max_turn
        self.extractor = TextExtractor()
        self.conv_key = conv_key
        self.turn_mode = turn_mode
        self.exp_name = exp_name

    def get_system_turn(self):
        return {
            "role": "system",
            "content": self.doctor_system_instruct
        }
    
    def finished_conversation(self, turn, doctor_reply):
        # if turn >= self.max_turn and self.max_turn != -1:
        #     return True
        if "preliminary_diagnoses" in doctor_reply.lower():
            return True
        if "preliminary diagnoses" in doctor_reply.lower():
            return True
        if ("{" in doctor_reply or "}" in doctor_reply or "[" in doctor_reply or "]" in doctor_reply):
            return True
        if 'disease:' in doctor_reply.lower():
            return True
        return False

    def save_finished_sample(self, sample, conversation, note_id, output_dir):
        try:
            ddx_obj = self.extractor.extract_first_json_object(conversation[-1]['content'])
        except Exception:
            ddx_obj = conversation[-1]['content']

        conv_inference = {
            "conversation": conversation[1:],
            "preliminary_diagnoses": ddx_obj
        }

        sample[self.conv_key] = conv_inference

        output_path = os.path.join(output_dir, f'{note_id}.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)

    def run_on_dataset_batch(self, dataset_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        conversation_history_doctor = []
        conversation_history_patient = []
        samples = []
        note_ids = []
        notes = []
        
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                if not file.endswith('.json'):
                    continue
                output_path = os.path.join(output_dir, file)
                if os.path.exists(output_path):
                    with open(output_path, 'r', encoding='utf-8') as f:
                        sample = json.load(f)
                        if self.conv_key in sample and self.finished_conversation(0, sample[self.conv_key]['conversation'][-1]['content']):
                            continue
                full_path = os.path.join(root, file)
                with open(full_path, 'r', encoding='utf-8') as f:
                    sample = json.load(f)

                samples.append(sample)

                note_id = sample['note_id']
                note_ids.append(note_id)

                hpi = sample['hpi']
                notes.append(hpi)
                patient_chief_complaint = sample['conv_revised']['conversation'][0]['content']
                
                conversation_doctor = [
                    {"role": "system", "content": self.doctor_system_instruct},
                    {"role": "user", "content": patient_chief_complaint}
                ]
                conversation_patient = [
                    {"role": "system", "content": get_patient_system_prompt(hpi)},
                    {"role": "assistant", "content": patient_chief_complaint}
                ]
                conversation_history_doctor.append(conversation_doctor)
                conversation_history_patient.append(conversation_patient)



        index = 1
        while conversation_history_doctor != []:
            index += 1
            print("turn: ", index)
            if index % 2 == 0:
                batch_doctor_replys = self.doctor_responder.generate_batch(conversation_history_doctor)
                indices_to_remove = []
                for i, doctor_reply in enumerate(batch_doctor_replys):
                    if i == 0:
                        print('doctor:', doctor_reply)
                    conversation_history_doctor[i].append({"role": "assistant", "content": doctor_reply})
                    next_action = doctor_reply.split('</think>')[-1]

                    conversation_history_patient[i].append({"role": "user", "content": next_action})
                    if self.finished_conversation(index, next_action):
                        self.save_finished_sample(
                            samples[i],
                            conversation_history_doctor[i],
                            note_ids[i],
                            output_dir
                        )
                        indices_to_remove.append(i)
                for i in reversed(indices_to_remove):
                    del conversation_history_doctor[i]
                    del conversation_history_patient[i]
                    del samples[i]
                    del note_ids[i]
                    del notes[i]
                if index >= self.max_turn:
                    break
            else:
                batch_patient_replys = self.patient_agent.generate_responses_in_batches(conversation_history_patient, max_token=150)

                for i, patient_reply in enumerate(batch_patient_replys):
                    if i == 0:
                        print('patient: ', patient_reply)
                    conversation_history_patient[i].append({"role": "assistant", "content": patient_reply})
                    conversation_history_doctor[i].append({"role": "user", "content": patient_reply})
        
        if conversation_history_doctor != [] and self.exp_name.startswith('zeroshot'):
            for i in range(len(conversation_history_doctor)):
                conversation_history_doctor[i][0]['content'] = get_doctor_system_prompt(version='ddx')
                conversation_history_doctor[i] = conversation_history_doctor[i][:-1]
            batch_doctor_replys = self.doctor_responder.generate_batch(conversation_history_doctor)
            for i, doctor_reply in enumerate(batch_doctor_replys):
                if i == 0:
                    print('doctor:', doctor_reply)
                conversation_history_doctor[i].append({"role": "assistant", "content": doctor_reply})
                self.save_finished_sample(
                    samples[i],
                    conversation_history_doctor[i],
                    note_ids[i],
                    output_dir
                )
        
        self.doctor_responder.close()

def build_simulator_gpt(model_name="gpt-4o",exp_name=None, **kwargs) -> SimulatorBase:
    return SimulatorBase(
        doctor_responder=GPTServerResponder(model_name),
        exp_name=exp_name,
        **kwargs
    )

def build_simulator_openllm(model_path, lora_path=None, exp_name=None, **kwargs) -> SimulatorBase:
    doctor_agent = DoctorAgent(model_path, lora_path, max_turn=kwargs.get('max_turn', 26), exp_name=exp_name) 
    return SimulatorBase(
        doctor_responder=LLMResponder(doctor_agent, turn_mode=kwargs.get("turn_mode", "multi")),
        exp_name=exp_name,
        **kwargs
    )