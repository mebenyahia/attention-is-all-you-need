import torch
from accelerate.utils import ProjectConfiguration
from torch.optim.lr_scheduler import ReduceLROnPlateau
from accelerate import Accelerator
from datasets import Dataset
from models import Transformer, DataloaderProvider
from tokenizers import Tokenizer
import json
import os
import config
from datasets import Dataset
from tokenizers import Tokenizer, trainers, models, pre_tokenizers, normalizers
import config
import os
import json
import math
from matplotlib import pyplot as plt

plt.ion()
resume = False

# only for mac, remove for training on cuda or cpu
if torch.backends.mps.is_available():
    os.environ["ACCELERATE_USE_MPS_DEVICE"] = "True"

print("Creating minimal training set for simplest model validation...")

sentences = [
    {"translation": {"en": "Hello, how are you?", "de": "Hallo, wie geht es dir?"}},
    {"translation": {"en": "Good morning!", "de": "Guten Morgen!"}},
    {"translation": {"en": "Good afternoon!", "de": "Guten Tag!"}},
    {"translation": {"en": "Good evening!", "de": "Guten Abend!"}},
    {"translation": {"en": "It's nice to see you.", "de": "Es ist schön, dich zu sehen."}},
    {"translation": {"en": "How have you been?", "de": "Wie ist es dir ergangen?"}},
    {"translation": {"en": "Long time no see!", "de": "Lange nicht gesehen!"}},
    {"translation": {"en": "What's new?", "de": "Was gibt's Neues?"}},
    {"translation": {"en": "How's your day going?", "de": "Wie läuft dein Tag?"}},
    {"translation": {"en": "I'm doing well, thanks.", "de": "Mir geht es gut, danke."}},
    {"translation": {"en": "Thank you so much.", "de": "Vielen Dank."}},
    {"translation": {"en": "I'm happy to hear that.", "de": "Das freut mich zu hören."}},
    {"translation": {"en": "What are you up to today?", "de": "Was hast du heute vor?"}},
    {"translation": {"en": "Have a great day!", "de": "Hab einen schönen Tag!"}},
    {"translation": {"en": "Take care.", "de": "Pass auf dich auf."}},
    {"translation": {"en": "See you soon.", "de": "Bis bald."}},
    {"translation": {"en": "Talk to you later.", "de": "Wir sprechen uns später."}},
    {"translation": {"en": "It's been a while.", "de": "Es ist schon eine Weile her."}},
    {"translation": {"en": "I've missed you.", "de": "Ich habe dich vermisst."}},
    {"translation": {"en": "How's your family?", "de": "Wie geht es deiner Familie?"}},
    {"translation": {"en": "What have you been doing lately?", "de": "Was hast du in letzter Zeit gemacht?"}},
    {"translation": {"en": "Any weekend plans?", "de": "Hast du Pläne für das Wochenende?"}},
    {"translation": {"en": "Did you sleep well?", "de": "Hast du gut geschlafen?"}},
    {"translation": {"en": "I hope your day is going well.", "de": "Ich hoffe, dein Tag läuft gut."}},
    {"translation": {"en": "Can I help you with anything?", "de": "Kann ich dir bei etwas helfen?"}},
    {"translation": {"en": "Enjoy your meal.", "de": "Guten Appetit."}},
    {"translation": {"en": "That sounds great.", "de": "Das klingt großartig."}},
    {"translation": {"en": "I'm looking forward to it.", "de": "Ich freue mich darauf."}},
    {"translation": {"en": "Do you have a minute?", "de": "Hast du eine Minute Zeit?"}},
    {"translation": {"en": "Everything's fine, thanks.", "de": "Alles in Ordnung, danke."}},
    {"translation": {"en": "I appreciate it.", "de": "Ich weiß das zu schätzen."}},
    {"translation": {"en": "Where are you heading?", "de": "Wohin gehst du?"}},
    {"translation": {"en": "What time is it?", "de": "Wie spät ist es?"}},
    {"translation": {"en": "I really like your outfit.", "de": "Ich mag dein Outfit wirklich sehr."}},
    {"translation": {"en": "You look great today.", "de": "Du siehst heute toll aus."}},
    {"translation": {"en": "Have you eaten yet?", "de": "Hast du schon gegessen?"}},
    {"translation": {"en": "I'm here if you need me.", "de": "Ich bin da, falls du mich brauchst."}},
    {"translation": {"en": "I'm glad to hear that.", "de": "Das freut mich."}},
    {"translation": {"en": "Sorry, I have to go.", "de": "Entschuldige, ich muss los."}},
    {"translation": {"en": "Can we talk later?", "de": "Können wir später reden?"}},
    {"translation": {"en": "Let me know if you need help.", "de": "Sag mir Bescheid, wenn du Hilfe brauchst."}},
    {"translation": {"en": "Would you like a coffee?", "de": "Möchtest du einen Kaffee?"}},
    {"translation": {"en": "It was nice talking to you.", "de": "Es war schön, mit dir zu reden."}},
    {"translation": {"en": "I'll see you tomorrow.", "de": "Wir sehen uns morgen."}},
    {"translation": {"en": "I hope you feel better soon.", "de": "Ich hoffe, es geht dir bald besser."}},
    {"translation": {"en": "Best of luck!", "de": "Viel Glück!"}},
    {"translation": {"en": "Don't worry about it.", "de": "Mach dir keine Sorgen."}},
    {"translation": {"en": "It happens to everyone.", "de": "Das passiert jedem."}},
    {"translation": {"en": "Thanks for your help.", "de": "Danke für deine Hilfe."}},
    {"translation": {"en": "Have a nice evening.", "de": "Schönen Abend noch."}}
]

train_dataset = Dataset.from_list(sentences)


print("Loading small tokenizer from saved...")
tokenizer = Tokenizer.from_file('tokenizer/bpe_small.json')

train_dlp = DataloaderProvider(train_dataset, 10, tokenizer)

output_dir = "saved_state"
accelerator_project_config = ProjectConfiguration(
    total_limit=3,
    automatic_checkpoint_naming=True,
    project_dir=output_dir,
    logging_dir=os.path.join(output_dir, 'logs'),
)

accelerator = Accelerator(project_config=accelerator_project_config)

print("Preparing model...")
model = Transformer(500, 128, 512, 4, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
loss_function = torch.nn.CrossEntropyLoss(ignore_index=train_dlp.pad)
sheduler = ReduceLROnPlateau(optimizer, factor=0.5)

model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dlp.dataloader)

if resume:
    accelerator.load_state("saved_state/final")

accelerator.register_for_checkpointing(sheduler)
accelerator.register_for_checkpointing(model)
accelerator.register_for_checkpointing(optimizer)


def train_epoch(model, train_dataloader, optimizer, loss_function, sheduler, accelerator):
    model.train()
    
    train_loss = 0
    num_batches = 0

    for batch in train_dataloader:

        sources = batch['sources']
        targets = batch['targets']

        optimizer.zero_grad()
        predictions = model(sources, targets[:, :-1])

        B, S, C = predictions.shape

        loss = loss_function(predictions.reshape(-1, C), targets[:, 1:].reshape(-1))
        accelerator.backward(loss)
        optimizer.step()

        #print(f'training loss: {loss.item()}')
        train_loss += loss.item()
        num_batches += 1

    avg_epoch_loss = train_loss/num_batches
    sheduler.step(avg_epoch_loss)
    return avg_epoch_loss


print("Training: ")
losses = []

for epoch in range(250):
    print(f'Epoch: {epoch}')
    loss = train_epoch(model=model, train_dataloader=train_dataloader, optimizer=optimizer, loss_function=loss_function,
                sheduler=sheduler, accelerator=accelerator)
    print(f'Average training loss: {loss}' )
    losses.append(loss)

plt.plot(range(250), losses)
plt.savefig('output.png')
print("Saving model...")
accelerator.save_model(model, "saved_model")
