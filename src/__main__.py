from llm_sdk import Small_LLM_Model
from src.vocabulary import Vocabulary, VocabularyError
from src.masking import mask_logits

llm = Small_LLM_Model()
prompt = "ciao"
encoded_prompt = llm.encode(prompt).squeeze(0).tolist()

try:
    # Load vocabulary to know the size and get text if needed
    logits = llm.get_logits_from_input_ids(encoded_prompt)
    vocab = Vocabulary.from_file(llm.get_path_to_vocab_file(), len(logits))

    # Efficient token-by-token generation loop
    for i in range(10):
        if i == 1:
            # Example of constraining allowed token IDs (e.g., only allow IDs 0, 1, 2)
            {}
        logits = llm.get_logits_from_input_ids(encoded_prompt)
        
        # Apply masking (currently allowing all IDs)
        allowed_ids = set(range(len(logits)))
        logits = mask_logits(logits, allowed_ids)
        
        # Pick best token and append to the ID list
        next_token = max(range(len(logits)), key=lambda i: logits[i])
        encoded_prompt.append(next_token)

    # Decode the full sequence at once
    print(llm.decode(encoded_prompt))

except VocabularyError as e:
    print(f"Error: {e}")