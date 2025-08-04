from Url_Project.URL_detection import classify_url_from_text
from Content_project.Context_detection import classify_context

def final_verdict(url_pred, context_pred):
    """Enhanced verdict system with more detailed outputs"""
    if not url_pred and not context_pred:
        return "ham"
    else:
        return "spam"
    
# This module combines URL and context classification to determine the final verdict of a message.
def classify_message(message: str) -> str:
    url_data = classify_url_from_text(message)
    
    url_result = 0  
    for url in url_data:
        if isinstance(url, dict):
            verdict = url.get('final_verdict', '').upper()
            if 'MALICIOUS' in verdict:
                url_result = 1 

    context_result = classify_context(message)
    return final_verdict(url_result, context_result)

def main():
    print("Enter messages (URLs or text) separated by semicolons:")
    user_input = input("Enter the messages: ")

    messages = [m.strip() for m in user_input.split(';') if m.strip()]
    
    for i, msg in enumerate(messages, 1):
        try:
            url_data = classify_url_from_text(msg)
            
            url_result = 0
            for url in url_data:
                if isinstance(url, dict):
                    verdict = url.get('final_verdict', '').upper()
                    if 'MALICIOUS' in verdict:
                        url_result = 1

            context_result = classify_context(msg)
            print(final_verdict(url_result, context_result))

        except Exception as e:
            print(f"Error processing message {i}: {str(e)}")

if __name__ == "__main__":
    main()
