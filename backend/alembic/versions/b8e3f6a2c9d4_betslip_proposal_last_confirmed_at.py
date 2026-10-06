"""add betting_slip_proposal_snapshots.last_confirmed_at

Revision ID: b8e3f6a2c9d4
Revises: f1a2b3c4d5e6
Create Date: 2026-10-06 00:00:00.000000

Distingue "confermata dal giro di generazione piu' recente" da "accumulata
da un giro precedente e mai piu' rigenerata" SENZA toccare `is_latest`
(che resta l'unico campo che conta per liquidazione/ROI - una proposta
salvata va sempre liquidata al suo esito reale, mai nascosta
retroattivamente). Vedi `BettingSlipProposalRepository.save_revision`.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b8e3f6a2c9d4"
down_revision: Union[str, Sequence[str], None] = "f1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "betting_slip_proposal_snapshots",
        sa.Column("last_confirmed_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("betting_slip_proposal_snapshots", "last_confirmed_at")
