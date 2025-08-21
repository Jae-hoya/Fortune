// Tarot card mapping and utility functions

export const TAROT_CARD_MAPPING: Record<string, string> = {
  "1": "The Fool",
  "2": "The Magician",
  "3": "The High Priestess",
  "4": "The Empress",
  "5": "The Emperor",
  "6": "The Hierophant",
  "7": "The Lovers",
  "8": "The Chariot",
  "9": "Strength",
  "10": "The Hermit",
  "11": "Wheel of Fortune",
  "12": "Justice",
  "13": "The Hanged Man",
  "14": "Death",
  "15": "Temperance",
  "16": "The Devil",
  "17": "The Tower",
  "18": "The Star",
  "19": "The Moon",
  "20": "The Sun",
  "21": "Judgement",
  "22": "The World",
  "23": "Ace of Cups",
  "24": "Two of Cups",
  "25": "Three of Cups",
  "26": "Four of Cups",
  "27": "Five of Cups",
  "28": "Six of Cups",
  "29": "Seven of Cups",
  "30": "Eight of Cups",
  "31": "Nine of Cups",
  "32": "Ten of Cups",
  "33": "Page of Cups",
  "34": "Knight of Cups",
  "35": "Queen of Cups",
  "36": "King of Cups",
  "37": "Ace of Pentacles",
  "38": "Two of Pentacles",
  "39": "Three of Pentacles",
  "40": "Four of Pentacles",
  "41": "Five of Pentacles",
  "42": "Six of Pentacles",
  "43": "Seven of Pentacles",
  "44": "Eight of Pentacles",
  "45": "Nine of Pentacles",
  "46": "Ten of Pentacles",
  "47": "Page of Pentacles",
  "48": "Knight of Pentacles",
  "49": "Queen of Pentacles",
  "50": "King of Pentacles",
  "51": "Ace of Swords",
  "52": "Two of Swords",
  "53": "Three of Swords",
  "54": "Four of Swords",
  "55": "Five of Swords",
  "56": "Six of Swords",
  "57": "Seven of Swords",
  "58": "Eight of Swords",
  "59": "Nine of Swords",
  "60": "Ten of Swords",
  "61": "Page of Swords",
  "62": "Knight of Swords",
  "63": "Queen of Swords",
  "64": "King of Swords",
  "65": "Ace of Wands",
  "66": "Two of Wands",
  "67": "Three of Wands",
  "68": "Four of Wands",
  "69": "Five of Wands",
  "70": "Six of Wands",
  "71": "Seven of Wands",
  "72": "Eight of Wands",
  "73": "Nine of Wands",
  "74": "Ten of Wands",
  "75": "Page of Wands",
  "76": "Knight of Wands",
  "77": "Queen of Wands",
  "78": "King of Wands"
}

// Convert card name to filename format
export function getCardImagePath(cardNumber: number): string {
  const cardName = TAROT_CARD_MAPPING[cardNumber.toString()]
  if (!cardName) {
    return "/tarot/cardback.png" // fallback to cardback
  }
  
  // Convert card name to filename format
  const filename = cardName.toLowerCase().replace(/\s+/g, '_')
  return `/tarot/tarot_cards/${cardNumber}_${filename}.png`
}

// Get card name from number
export function getCardName(cardNumber: number): string {
  return TAROT_CARD_MAPPING[cardNumber.toString()] || `Card ${cardNumber}`
} 