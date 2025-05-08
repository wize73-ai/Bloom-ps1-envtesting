#!/bin/bash
# Test translation quality with the upgraded models

# Set base URL - default to localhost:8000 if not provided
BASE_URL=${1:-http://localhost:8000}
ENDPOINT="/pipeline/translate"
echo "Testing translation quality at $BASE_URL$ENDPOINT"

# Function to make a translation request and display results
test_translation() {
  local text=$1
  local source=$2
  local target=$3
  local description=$4
  
  echo -e "\n=== Testing $description ==="
  echo "Original ($source): \"$text\""
  
  # Create JSON payload
  payload="{\"text\":\"$text\",\"source_language\":\"$source\",\"target_language\":\"$target\"}"
  
  # Make the request
  response=$(curl -s -X POST "$BASE_URL$ENDPOINT" \
       -H "Content-Type: application/json" \
       -d "$payload")
  
  # Extract translated text
  translated=$(echo "$response" | grep -o '"translated_text":"[^"]*"' | sed 's/"translated_text":"\(.*\)"/\1/')
  
  echo "Translated ($target): \"$translated\""
}

# Test cases designed to evaluate translation quality
test_translation "The quick brown fox jumps over the lazy dog." "en" "es" "Simple sentence (EN->ES)"
test_translation "El zorro marrón rápido salta sobre el perro perezoso." "es" "en" "Simple sentence (ES->EN)"

test_translation "The new machine learning models demonstrate unprecedented levels of accuracy when trained on large, diverse datasets." "en" "fr" "Technical content (EN->FR)"
test_translation "Les nouveaux modèles d'apprentissage automatique démontrent des niveaux de précision sans précédent lorsqu'ils sont entraînés sur des ensembles de données volumineux et diversifiés." "fr" "en" "Technical content (FR->EN)"

test_translation "Despite the challenges, the team managed to complete the project ahead of schedule and under budget, which impressed the stakeholders." "en" "de" "Complex sentence (EN->DE)"
test_translation "Trotz der Herausforderungen gelang es dem Team, das Projekt vor dem Zeitplan und unter dem Budget abzuschließen, was die Stakeholder beeindruckte." "de" "en" "Complex sentence (DE->EN)"

test_translation "Her smile was as bright as the morning sun, warming everyone's hearts in the room." "en" "it" "Figurative language (EN->IT)"
test_translation "Il suo sorriso era luminoso come il sole del mattino, riscaldando i cuori di tutti nella stanza." "it" "en" "Figurative language (IT->EN)"

echo -e "\n=== Translation quality tests completed ==="