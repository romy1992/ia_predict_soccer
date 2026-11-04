"""ledger_p_adjusted

Revision ID: b9c8d7e6f5a4
Revises: e4f5a6b7c8d9
Create Date: 2026-11-04 00:00:00.000000

Prediction ledger: probabilita' usata per EV/PLAY dopo la riduzione verso il
mercato (policy v3) e peso del modello. Colonne nuove, nullable, additive:
le righe esistenti restano invariate (p_model e' la probabilita' grezza).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'b9c8d7e6f5a4'
down_revision: Union[str, Sequence[str], None] = 'e4f5a6b7c8d9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('prediction_ledger', sa.Column('p_adjusted', sa.Float(), nullable=True))
    op.add_column('prediction_ledger', sa.Column('model_weight', sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column('prediction_ledger', 'model_weight')
    op.drop_column('prediction_ledger', 'p_adjusted')
