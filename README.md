# DeductionNetwork
演驛のようなことができるニューラルネットワーク
TransformerのEncoder部分から、PositionalEncodingを取り除き、Self-AttensionではなくQuery,Key,Valueにそれぞれ異なる入力を行う。 

ネットワーク内部にQuery,Key,Valueに対応するTableを保持しておき、更新する。 

正直コード見たほうが早い。
