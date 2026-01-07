import os
from datetime import datetime, timedelta

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from dotenv import load_dotenv
import stripe

load_dotenv(dotenv_path='../../../properties/config.env')
TOKEN = os.environ.get("TELEGRAM_TOKEN")
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")
WEEKLY_LIMIT_EUR = 50
user_payments = {
    'user_id': [
        {"amount": 1.99, "date": datetime}
    ]
}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Ciao! Premi /paga per effettuare un pagamento."
    )


def get_weekly_total(user_id):
    now = datetime.now()
    week_ago = now - timedelta(days=7)

    if user_id not in user_payments:
        return 0

    return sum(
        p["amount"]
        for p in user_payments[user_id]
        if p["date"] >= week_ago
    )


async def paga(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_id = update.effective_user.id
        amount_eur = 1.99

        if amount_eur <= 0:
            raise ValueError

        spent_this_week = get_weekly_total(user_id)

        if spent_this_week + amount_eur > WEEKLY_LIMIT_EUR:
            await update.message.reply_text(
                f"❌ Limite settimanale superato.\n"
                f"Hai già speso {spent_this_week}€ su {WEEKLY_LIMIT_EUR}€ disponibili."
            )
            return

        # Stripe vuole i centesimi
        amount = amount_eur * 100

        payment_intent = stripe.PaymentIntent.create(
            amount=amount,
            currency="eur",
            payment_method_types=["card"],
            metadata={
                "telegram_user_id": user_id
            }
        )

        # salva pagamento "pending"
        user_payments.setdefault(user_id, []).append({
            "amount": amount_eur,
            "date": datetime.now()
        })

        await update.message.reply_text(
            f"✅ Pagamento avviato ({amount_eur}€).\n"
            f"Totale settimanale: {spent_this_week + amount_eur}€ / {WEEKLY_LIMIT_EUR}€\n\n"
            f"client_secret:\n{payment_intent.client_secret}"
        )

    except (IndexError, ValueError):
        await update.message.reply_text(
            "Importo non valido. Usa ad esempio:\n/paga 10"
        )
    except Exception as e:
        await update.message.reply_text("Errore durante il pagamento.")
        print(e)


app = ApplicationBuilder().token(TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("paga", paga))

app.run_polling()
