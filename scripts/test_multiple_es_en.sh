#!/bin/bash
# Test multiple Spanish to English translations

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Server URL (default to localhost:8000 but can be overridden)
SERVER_URL=${1:-"http://localhost:8000"}
ENDPOINT="/pipeline/translate"

echo -e "${BLUE}Testing Multiple Spanish to English Translations${NC}"
echo -e "${YELLOW}Server:${NC} $SERVER_URL$ENDPOINT"

# Test sentences
declare -a TEST_SENTENCES=(
  "Estoy muy feliz de conocerte hoy. El clima es hermoso y espero que estés bien."
  "Hola, ¿cómo estás?"
  "Me gusta aprender idiomas nuevos."
  "El español es un idioma hermoso con muchos hablantes en todo el mundo."
  "Necesito ayuda con mi tarea de matemáticas, por favor."
  "El fin de semana voy a visitar a mi familia en Barcelona."
  "Los gatos son animales muy interesantes y curiosos."
  "¿Puedes recomendarme un buen restaurante en la ciudad?"
  "Me encantaría viajar por Sudamérica para conocer diferentes culturas."
  "La inteligencia artificial está cambiando la forma en que interactuamos con la tecnología."
)

# Czech words to check for
declare -a CZECH_WORDS=("Jsem" "velmi" "že" "vás" "poznávám" "dnes" "šťastný" "rád" "Těší")

# Loop through and test each sentence
for ((i=0; i<${#TEST_SENTENCES[@]}; i++)); do
  SENTENCE="${TEST_SENTENCES[$i]}"
  echo -e "\n${YELLOW}Test #$((i+1)):${NC} $SENTENCE"
  
  # Prepare JSON data for the request
  JSON_DATA=$(cat <<EOF
{
  "text": "$SENTENCE",
  "source_language": "es",
  "target_language": "en"
}
EOF
)

  # Send request to the translation endpoint
  echo -e "${YELLOW}Sending request...${NC}"
  RESULT=$(curl -s -X POST \
    "$SERVER_URL$ENDPOINT" \
    -H "Content-Type: application/json" \
    -d "$JSON_DATA")
  
  # Extract translated text
  TRANSLATED_TEXT=$(echo $RESULT | grep -o '"translated_text":"[^"]*"' | sed 's/"translated_text":"//g' | sed 's/"//g')
  
  echo -e "${GREEN}Translation:${NC} $TRANSLATED_TEXT"
  
  # Check for Czech words
  HAS_CZECH=false
  for word in "${CZECH_WORDS[@]}"; do
    if echo "$TRANSLATED_TEXT" | grep -q "$word"; then
      echo -e "${RED}Found Czech word: $word${NC}"
      HAS_CZECH=true
    fi
  done
  
  if [ "$HAS_CZECH" = false ]; then
    echo -e "${GREEN}SUCCESS: No Czech words detected${NC}"
  fi
  
  # Check if translation is empty
  if [ -z "$TRANSLATED_TEXT" ]; then
    echo -e "${RED}ERROR: Empty translation${NC}"
  fi
  
  # Add separator
  echo -e "${BLUE}----------------------------------------${NC}"
done

echo -e "\n${BLUE}All tests completed!${NC}"