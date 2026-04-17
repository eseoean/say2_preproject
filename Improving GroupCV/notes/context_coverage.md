# Context Coverage

현재 `context_categorical` variant는 hybrid sample-aware cache에서 categorical/context 정보를 가져와 현재 common input에 붙이는 방식이다.

매칭 키

- common input:
  - `sample_id`
  - `canonical_drug_id`
- context source:
  - `CELL_LINE_NAME`
  - `DRUG_ID`

현재 확인된 coverage

- current common rows: `7730`
- matched rows: `3295`
- unmatched rows: `4435`
- match rate: `42.63%`

해석

- cell line overlap은 충분하지만 drug overlap이 제한적이다.
- 즉, 현재 common input의 `295`개 drug 중 hybrid cache에서 categorical/context source를 바로 제공하는 drug는 `125`개 수준이다.
- 따라서 이번 실험의 첫 단계는 "전체 row에 풍부한 context를 붙인다"기보다,
  "붙일 수 있는 context는 붙이고 나머지는 `__MISSING__` category로 유지했을 때도 GroupCV가 개선되는지"를 보는 형태가 된다.

주의

- 이 coverage 제한 때문에 초기 variant 성능이 크게 개선되지 않을 수도 있다.
- 만약 효과가 약하면 다음 단계에서는 현재 파이프라인 데이터에서 직접 복원 가능한 drug context를 추가로 만들어 coverage를 넓혀야 한다.
