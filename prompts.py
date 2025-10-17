api_prompt = {
"system_prompt":
"You are an assistant that answers questions about categories at the same level as input categories.\
\nWhen the user asks a question like 'What is the category at the same level as [Input Category]?', \
you should provide another category at the same level as the input category.\
\nAlways answer in the format:\
\nA: [Category] is the category at the same level as [Input Category]."
,
"user_prompt":
"\nQ: What is the category at the same level as "
,
"Caltech101":
"Q: What is the category at the same level as camera?\
\nA: Laptop is the category at the same level as camera.\
\nQ: What is the category at the same level as elephant?\
\nA: Panda is the category at the same level as elephant."
,
"DescribableTextures":
"Q: What is the category at the same level as banded?\
\nA: Striped is the category at the same level as banded.\
\nQ: What is the category at the same level as crystalline?\
\nA: Fibrous is the category at the same level as crystalline."
,
"EuroSAT":
"Q: What is the category at the same level as Annual Crop Land?\
\nA: Permanent Crop Land is the category at the same level as Annual Crop Land.\
\nQ: What is the category at the same level as Industrial Buildings?\
\nA: Residential Buildings is the category at the same level as Industrial Buildings."
,
"FGVCAircraft":
"Q: What is the category at the same level as 707-320?\
\nA: Falcon 2000 is the category at the same level as 707-320.\
\nQ: What is the category at the same level as A310?\
\nA: Saab 2000 is the category at the same level as A310."
,
"Food101":
"Q: What is the category at the same level as apple_pie?\
\nA: Hot_dog is the category at the same level as apple_pie.\
\nQ: What is the category at the same level as dumplings?\
\nA: Gyoza is the category at the same level as dumplings."
,
"ImageNet":
"Q: What is the category at the same level as basketball?\
\nA: Tennis ball is the category at the same level as basketball.\
\nQ: What is the category at the same level as mobile phone?\
\nA: Desktop computer is the category at the same level mobile phone."
,
"OxfordFlowers":
"Q: What is the category at the same level as carnation?\
\nA: Chrysanthemum is the category at the same level as carnation.\
\nQ: What is the category at the same level as love in the mist?\
\nA: Forget me not is the category at the same level as love in the mist."
,
"OxfordPets":
"Q: What is the category at the same level as birman?\
\nA: Siamese is the category at the same level as birman.\
\nQ: What is the category at the same level as american_pit_bull_terrier?\
\nA: Staffordshire_bull_terrier is the category at the same level as american_pit_bull_terrier."
,
"StanfordCars":
"Q: What is the category at the same level as 2012 BMW X6 SUV?\
\nA: 2009 Mercedes-Benz SL-Class Coupe is the category at the same level as 2012 BMW X6 SUV.\
\nQ: What is the category at the same level as 2011 Bentley Mulsanne Sedan?\
\nA: 2012 Ferrari California Convertible is the category at the same level as 2011 Bentley Mulsanne Sedan."
,
"SUN397":
"Q: What is the category at the same level as abbey?\
\nA: Monastery is the category at the same level as abbey.\
\nQ: What is the category at the same level as boxing_ring?\
\nA: Wrestling_ring is the category at the same level as boxing_ring."
,
"UCF101":
"Q: What is the category at the same level as Archery?\
\nA: Fencing is the category at the same level as Archery.\
\nQ: What is the category at the same level as Basketball_Dunk?\
\nA: Volleyball_Spiking is the category at the same level as Basketball_Dunk."
}

llama_prompt={
"system_prompt":
"As a caption summarizer, your task is to transform the provided captions from their original category to a new specified category \
and condense them into a concise set of 3 distinct one-sentence captions. \
Make sure the new captions maintain coherence with the original style but reflect the characteristics of the new target category. \
Each caption must capture a unique artistic style or visual theme. \
Only generate the transformed one-sentence captions—no introductions, explanations, or comments. \
The output should strictly follow this format:\n\
1. [Caption 1]\n\
2. [Caption 2]\n\
3. [Caption 3]"
,
"user_prompt":
['Transform and condense the following captions into 3 new one-sentence captions describing ', ', each focusing on a distinct artistic style or visual theme.']
}