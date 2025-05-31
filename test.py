from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

class QuestionGenerator:
    def __init__(self, model_name="./flan-t5-base"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
    def generate_questions_from_context(self, context, num_questions=8):
        questions = []
        prompts = [
            f"Generate a factual question about this text: {context}",
            f"What analytical question can be asked about: {context}",  
            f"Create a specific question based on: {context}",
            f"What detailed question relates to: {context}",
            f"Form a comprehensive question about: {context}",
            f"What application question can be derived from: {context}",
            f"Generate a comparison question about: {context}",
            f"What cause-and-effect question can be asked about: {context}"
        ]
        
        for i in range(num_questions):
            input_text = prompts[i % len(prompts)]
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, 
                                  truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=90,  
                    min_length=15,  
                    num_beams=6,    
                    num_return_sequences=1,
                    temperature=0.7 + (i * 0.05),  
                    do_sample=True,
                    early_stopping=True,
                    no_repeat_ngram_size=3,  
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            question = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            questions.append({
                    'id': len(questions) + 1,
                    'question': question
                })
        
        return questions
    

def generate_questions_from_context(context, num_questions=5):
    generator = QuestionGenerator()
    return generator.generate_questions_from_context(context, num_questions)

if __name__ == "__main__":
    test_context = """
    Artificial Intelligence (AI) has revolutionized numerous industries through machine learning algorithms and neural networks. 
    Deep learning, a subset of machine learning, utilizes multilayered neural networks to process vast amounts of data and identify 
    complex patterns. Convolutional Neural Networks (CNNs) excel in image recognition tasks, while Recurrent Neural Networks (RNNs) 
    and their advanced variants like Long Short-Term Memory (LSTM) networks are particularly effective for sequential data processing 
    such as natural language processing and time series analysis.
    """

    print("=" * 60)
    
    questions = generate_questions_from_context(test_context, num_questions=8)
    
    for q in questions:
        print(f"{q['id']}. {q['question']}")
