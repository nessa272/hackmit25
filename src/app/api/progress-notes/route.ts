import Cerebras from '@cerebras/cerebras_cloud_sdk';
import { NextRequest, NextResponse } from 'next/server';

const cerebras = new Cerebras({
  apiKey: process.env.CEREBRAS_API_KEY
});

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }

    let text: string;
    if (file.type === 'application/pdf') {
      const PDFParser = (await import('pdf2json')).default;
      const pdfParser = new PDFParser();
      
      const buffer = await file.arrayBuffer();
      
      const pdfText = await new Promise<string>((resolve, reject) => {
        pdfParser.on('pdfParser_dataError', reject);
        pdfParser.on('pdfParser_dataReady', (pdfData: any) => {
          let extractedText = '';
          pdfData.Pages.forEach((page: any) => {
            page.Texts.forEach((textObj: any) => {
              extractedText += decodeURIComponent(textObj.R[0].T) + ' ';
            });
          });
          resolve(extractedText);
        });
        pdfParser.parseBuffer(Buffer.from(buffer));
      });
      
      text = pdfText;
    } else {
      text = await file.text();
    }
    
    const completion = await cerebras.chat.completions.create({
      messages: [
        {
          role: "system",
          content: "Analyze this progress note and provide a summary in paragraph form only. Output plain text without any formatting, bullet points, lists, or special characters. Focus on patient status, care plan updates, and key medical observations."
        },
        {
          role: "user", 
          content: text
        }
      ],
      model: 'gpt-oss-120b',
      stream: false,
      max_completion_tokens: 2000,
      temperature: 1,
      top_p: 1,
      reasoning_effort: "medium"
    });

    return NextResponse.json({ 
      analysis: (completion as any).choices[0].message.content,
      type: 'Progress Notes'
    });
  } catch (error) {
    console.log(error);
    return NextResponse.json({ error: 'Analysis failed' }, { status: 500 });
  }
}
