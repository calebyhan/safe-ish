import { NextResponse } from 'next/server';
import { getPools } from '@/lib/db';

export async function GET() {
  try {
    const pools = getPools();
    return NextResponse.json({ pools });
  } catch (error) {
    console.error('Error fetching pools:', error);
    return NextResponse.json(
      { error: 'Failed to fetch pools' },
      { status: 500 }
    );
  }
}
