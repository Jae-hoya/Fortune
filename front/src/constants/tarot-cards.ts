export interface TarotCard {
  id: number
  name: string
  nameKo: string
  suit: string
  description: string
  imageUrl: string
  keywords: string[]
}

// 메이저 아르카나 예시 (실제로는 78장 전체를 구현해야 합니다)
export const MAJOR_ARCANA: TarotCard[] = [
  {
    id: 0,
    name: "The Fool",
    nameKo: "바보",
    suit: "Major Arcana",
    description: "새로운 시작, 순수함, 모험, 자유로움을 상징합니다. 때로는 무모함이나 위험을 나타내기도 합니다.",
    imageUrl: "/tarot/major/00.jpg",
    keywords: ["새로운 시작", "모험", "순수함", "자유", "무모함"]
  },
  {
    id: 1,
    name: "The Magician",
    nameKo: "마법사",
    suit: "Major Arcana",
    description: "창의력, 숙련된 기술, 자신감, 의지력을 상징합니다. 당신의 잠재력을 실현할 때입니다.",
    imageUrl: "/tarot/major/01.jpg",
    keywords: ["창의력", "기술", "자신감", "의지력", "실현"]
  },
  {
    id: 2,
    name: "The High Priestess",
    nameKo: "여사제",
    suit: "Major Arcana",
    description: "직관, 비밀, 내면의 지혜, 무의식을 상징합니다. 보이는 것 이면의 진실을 봐야 합니다.",
    imageUrl: "/tarot/major/02.jpg",
    keywords: ["직관", "비밀", "지혜", "무의식"]
  },
  {
    id: 3,
    name: "The Empress",
    nameKo: "여황제",
    suit: "Major Arcana",
    description: "풍요, 다산, 모성애, 아름다움, 자연을 상징합니다. 창조적인 에너지가 넘치는 시기입니다.",
    imageUrl: "/tarot/major/03.jpg",
    keywords: ["풍요", "다산", "모성애", "자연"]
  },
  {
    id: 4,
    name: "The Emperor",
    nameKo: "황제",
    suit: "Major Arcana",
    description: "권위, 구조, 통제, 아버지의 모습을 상징합니다. 안정과 질서를 세워야 할 때입니다.",
    imageUrl: "/tarot/major/04.jpg",
    keywords: ["권위", "구조", "통제", "안정"]
  },
  {
    id: 5,
    name: "The Hierophant",
    nameKo: "교황",
    suit: "Major Arcana",
    description: "전통, 종교, 신념, 교육, 제도를 상징합니다. 기존의 가치관과 규칙을 따르는 것이 중요합니다.",
    imageUrl: "/tarot/major/05.jpg",
    keywords: ["전통", "신념", "교육", "제도"]
  },
  {
    id: 6,
    name: "The Lovers",
    nameKo: "연인",
    suit: "Major Arcana",
    description: "사랑, 관계, 조화, 선택을 상징합니다. 중요한 결정을 내려야 할 수 있습니다.",
    imageUrl: "/tarot/major/06.jpg",
    keywords: ["사랑", "관계", "조화", "선택"]
  },
  {
    id: 7,
    name: "The Chariot",
    nameKo: "전차",
    suit: "Major Arcana",
    description: "승리, 의지력, 자기 통제, 진전을 상징합니다. 목표를 향해 단호하게 나아가야 합니다.",
    imageUrl: "/tarot/major/07.jpg",
    keywords: ["승리", "의지력", "전진"]
  },
  {
    id: 8,
    name: "Strength",
    nameKo: "힘",
    suit: "Major Arcana",
    description: "내면의 힘, 용기, 인내, 동정심을 상징합니다. 부드러움이 강함을 이길 수 있습니다.",
    imageUrl: "/tarot/major/08.jpg",
    keywords: ["용기", "인내", "내면의 힘"]
  },
  {
    id: 9,
    name: "The Hermit",
    nameKo: "은둔자",
    suit: "Major Arcana",
    description: "성찰, 내면 탐구, 지혜, 고독을 상징합니다. 진리를 찾기 위해 내면으로 향해야 합니다.",
    imageUrl: "/tarot/major/09.jpg",
    keywords: ["성찰", "탐구", "고독", "지혜"]
  },
  {
    id: 10,
    name: "Wheel of Fortune",
    nameKo: "운명의 수레바퀴",
    suit: "Major Arcana",
    description: "운명, 변화, 전환점, 행운을 상징합니다. 삶의 순환과 변화를 받아들여야 합니다.",
    imageUrl: "/tarot/major/10.jpg",
    keywords: ["운명", "변화", "순환", "행운"]
  },
  {
    id: 11,
    name: "Justice",
    nameKo: "정의",
    suit: "Major Arcana",
    description: "공정, 균형, 진실, 법, 원인과 결과를 상징합니다. 모든 행동에는 결과가 따릅니다.",
    imageUrl: "/tarot/major/11.jpg",
    keywords: ["공정", "균형", "진실", "법"]
  },
  {
    id: 12,
    name: "The Hanged Man",
    nameKo: "매달린 남자",
    suit: "Major Arcana",
    description: "희생, 새로운 관점, 정지, 순응을 상징합니다. 상황을 다르게 보아야 할 필요가 있습니다.",
    imageUrl: "/tarot/major/12.jpg",
    keywords: ["희생", "관점", "정지", "순응"]
  },
  {
    id: 13,
    name: "Death",
    nameKo: "죽음",
    suit: "Major Arcana",
    description: "끝, 변화, 전환, 새로운 시작을 상징합니다. 물리적 죽음보다는 변화의 의미가 강합니다.",
    imageUrl: "/tarot/major/13.jpg",
    keywords: ["끝", "변화", "전환"]
  },
  {
    id: 14,
    name: "Temperance",
    nameKo: "절제",
    suit: "Major Arcana",
    description: "균형, 조화, 인내, 중용을 상징합니다. 극단을 피하고 조화를 찾아야 합니다.",
    imageUrl: "/tarot/major/14.jpg",
    keywords: ["균형", "조화", "인내", "중용"]
  },
  {
    id: 15,
    name: "The Devil",
    nameKo: "악마",
    suit: "Major Arcana",
    description: "속박, 중독, 물질주의, 부정적인 패턴을 상징합니다. 자신을 묶고 있는 것을 깨달아야 합니다.",
    imageUrl: "/tarot/major/15.jpg",
    keywords: ["속박", "중독", "물질주의"]
  },
  {
    id: 16,
    name: "The Tower",
    nameKo: "탑",
    suit: "Major Arcana",
    description: "급작스러운 변화, 파괴, 계시, 각성을 상징합니다. 기존의 구조가 무너지고 새로운 것이 시작됩니다.",
    imageUrl: "/tarot/major/16.jpg",
    keywords: ["급격한 변화", "파괴", "계시"]
  },
  {
    id: 17,
    name: "The Star",
    nameKo: "별",
    suit: "Major Arcana",
    description: "희망, 영감, 치유, 긍정을 상징합니다. 어려운 시기 후의 평온과 희망을 의미합니다.",
    imageUrl: "/tarot/major/17.jpg",
    keywords: ["희망", "영감", "치유", "긍정"]
  },
  {
    id: 18,
    name: "The Moon",
    nameKo: "달",
    suit: "Major Arcana",
    description: "불안, 환상, 직관, 무의식을 상징합니다. 모든 것이 명확하지 않을 수 있으니 직관을 믿어야 합니다.",
    imageUrl: "/tarot/major/18.jpg",
    keywords: ["불안", "환상", "직관"]
  },
  {
    id: 19,
    name: "The Sun",
    nameKo: "태양",
    suit: "Major Arcana",
    description: "성공, 활기, 기쁨, 명확성을 상징합니다. 긍정적인 에너지와 성공이 함께합니다.",
    imageUrl: "/tarot/major/19.jpg",
    keywords: ["성공", "활기", "기쁨", "명확성"]
  },
  {
    id: 20,
    name: "Judgement",
    nameKo: "심판",
    suit: "Major Arcana",
    description: "부활, 심판, 용서, 새로운 시작을 상징합니다. 과거를 평가하고 새로운 단계로 나아갈 때입니다.",
    imageUrl: "/tarot/major/20.jpg",
    keywords: ["부활", "평가", "용서"]
  },
  {
    id: 21,
    name: "The World",
    nameKo: "세계",
    suit: "Major Arcana",
    description: "완성, 통합, 성취, 여행을 상징합니다. 하나의 주기가 끝나고 성공적으로 마무리됩니다.",
    imageUrl: "/tarot/major/21.jpg",
    keywords: ["완성", "성취", "통합", "여행"]
  }
]

export const WANDS: TarotCard[] = [
    { id: 22, name: "Ace of Wands", nameKo: "완드 에이스", suit: "Wands", description: "새로운 시작, 영감, 창의력, 잠재력", imageUrl: "/tarot/wands/w01.jpg", keywords: ["새로운 시작", "영감", "창의력"] },
    { id: 23, name: "Two of Wands", nameKo: "완드 2", suit: "Wands", description: "계획, 결정, 미래 준비, 파트너십", imageUrl: "/tarot/wands/w02.jpg", keywords: ["계획", "결정", "미래"] },
    { id: 24, name: "Three of Wands", nameKo: "완드 3", suit: "Wands", description: "확장, 해외, 장기적인 비전, 기다림", imageUrl: "/tarot/wands/w03.jpg", keywords: ["확장", "비전", "기다림"] },
    { id: 25, name: "Four of Wands", nameKo: "완드 4", suit: "Wands", description: "축하, 안정, 가정, 조화", imageUrl: "/tarot/wands/w04.jpg", keywords: ["축하", "안정", "가정"] },
    { id: 26, name: "Five of Wands", nameKo: "완드 5", suit: "Wands", description: "경쟁, 갈등, 사소한 다툼, 도전", imageUrl: "/tarot/wands/w05.jpg", keywords: ["경쟁", "갈등", "도전"] },
    { id: 27, name: "Six of Wands", nameKo: "완드 6", suit: "Wands", description: "승리, 성공, 인정, 대중의 지지", imageUrl: "/tarot/wands/w06.jpg", keywords: ["승리", "성공", "인정"] },
    { id: 28, name: "Seven of Wands", nameKo: "완드 7", suit: "Wands", description: "방어, 도전, 용기, 자신의 입장을 지키기", imageUrl: "/tarot/wands/w07.jpg", keywords: ["방어", "도전", "용기"] },
    { id: 29, name: "Eight of Wands", nameKo: "완드 8", suit: "Wands", description: "빠른 진행, 행동, 소식, 여행", imageUrl: "/tarot/wands/w08.jpg", keywords: ["빠른 진행", "행동", "소식"] },
    { id: 30, name: "Nine of Wands", nameKo: "완드 9", suit: "Wands", description: "인내, 방어, 마지막 저항, 회복력", imageUrl: "/tarot/wands/w09.jpg", keywords: ["인내", "방어", "회복력"] },
    { id: 31, name: "Ten of Wands", nameKo: "완드 10", suit: "Wands", description: "부담, 책임감, 압박감, 과로", imageUrl: "/tarot/wands/w10.jpg", keywords: ["부담", "책임감", "과로"] },
    { id: 32, name: "Page of Wands", nameKo: "완드 페이지", suit: "Wands", description: "열정적인 소식, 새로운 아이디어, 탐험, 자유로운 영혼", imageUrl: "/tarot/wands/w11.jpg", keywords: ["새로운 아이디어", "탐험", "열정"] },
    { id: 33, name: "Knight of Wands", nameKo: "완드 나이트", suit: "Wands", description: "에너지, 열정, 행동, 충동적인 여행", imageUrl: "/tarot/wands/w12.jpg", keywords: ["에너지", "행동", "충동"] },
    { id: 34, name: "Queen of Wands", nameKo: "완드 퀸", suit: "Wands", description: "자신감, 독립성, 열정, 카리스마", imageUrl: "/tarot/wands/w13.jpg", keywords: ["자신감", "독립성", "카리스마"] },
    { id: 35, name: "King of Wands", nameKo: "완드 킹", suit: "Wands", description: "리더십, 비전, 카리스마, 기업가 정신", imageUrl: "/tarot/wands/w14.jpg", keywords: ["리더십", "비전", "기업가 정신"] }
];

export const FULL_DECK: TarotCard[] = [
  ...MAJOR_ARCANA,
  ...WANDS,
]

export const SPREADS = {
  threeCard: {
    name: "3장 스프레드",
    description: "과거, 현재, 미래를 보는 기본적인 스프레드입니다.",
    positions: [
      { name: "과거", description: "지난 일들의 영향" },
      { name: "현재", description: "현재 상황" },
      { name: "미래", description: "앞으로의 방향" }
    ]
  },
  celtic: {
    name: "켈틱 크로스",
    description: "상황을 더 깊이 있게 분석하는 10장 스프레드입니다.",
    positions: [
      { name: "현재 상황", description: "현재의 핵심 이슈" },
      { name: "당면 과제", description: "현재 직면한 문제" },
      { name: "과거의 기반", description: "과거로부터의 영향" },
      { name: "최근 과거", description: "최근에 일어난 일" },
      { name: "가능한 결과", description: "잠재적 결과" },
      { name: "가까운 미래", description: "곧 다가올 일" },
      { name: "현재 태도", description: "자신의 태도" },
      { name: "외부 영향", description: "주변 환경의 영향" },
      { name: "희망과 두려움", description: "내면의 감정" },
      { name: "최종 결과", description: "최종적인 결과" }
    ]
  }
} 