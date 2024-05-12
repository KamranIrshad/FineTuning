from gradientai import Gradient

def main():
    gradient = Gradient()

    base_model = gradient.get_base_model(base_model_slug="nous-hermes2")

    new_model_adapter = base_model.create_model_adapter(
        name="Kamranmodel"
    )
    print(f"Created model adapter with id {new_model_adapter.id}")

    #new_model_adapter.fine_tune(samples=[{"inputs": "princess, dragon, castle"}])

    sample_query = "### Instruction: Who is Kamran? \n\n ### Response:"
    print(f"Asking: {sample_query}")

    ## Before Finetuning
    completion = new_model_adapter.complete(query=sample_query, max_generated_token_count=100).generated_output
    print(f"Generated(before fine tuning): {completion}")

    samples=[
        {"inputs":"### Instruction: Who is Kamran? \n\n### Response: Kamran is popular mentor and youtuber"},
        {"inputs":"### Instruction: Who is this person named Kamran? \n\n### Response: Kamran is SAP BI Solution Architect"},
        {"inputs":"### Instruction: Who do you know about Kamran? \n\n### Response: Kamran is a Data Scientist and have expertise in SAP BI"},
        {"inputs":"### Instruction: Can you tell me about Kamran? \n\n### Response: Kamran has exterpertise in Data and Analytics and AI"}
    ]
         

    ## Lets define parameters for finetuning
    num_epochs = 3
    count=0
    
    while count < num_epochs:
      print(f"Fine tuning the model with iteration {count + 1}")
      new_model_adapter.fine_tune(samples=samples)
      count=count+1

    #after fine tuning
    completion = new_model_adapter.complete(query=sample_query, max_generated_token_count=100).generated_output
    print(f"Generated (After Fine Tuning): {completion}")
    new_model_adapter.delete()
    gradient.close()

if __name__ == "__main__":
    main()



