import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Scanner;
import java.util.StringTokenizer;
import java.util.Map.Entry;

import org.fnlp.app.keyword.AbstractExtractor;
import org.fnlp.app.keyword.WordExtract;

import edu.fudan.ml.types.Dictionary;
import edu.fudan.nlp.cn.tag.CWSTagger;
import edu.fudan.nlp.corpus.StopWords;
import edu.fudan.util.exception.LoadModelException;


public class ExtractKeyWords {

	/**
	 * @param args
	 * @throws UnsupportedEncodingException
	 */
	private static CWSTagger seg;
	private static StopWords sw;
	private static HashMap<String,Double> idf = new HashMap<String,Double>();
	public static ArrayList<String> userdictionary = new ArrayList<String>();//�û��Զ���ʵ�
	public static HashSet<String> userdictionaryset = new HashSet<String>();
	public ExtractKeyWords() throws IOException, Exception
	{
		seg = new CWSTagger("H:\\Coding Toolkits\\FudanNLP-1.6.1\\models\\seg.m",new Dictionary("H:\\Coding Toolkits\\FudanNLP-1.6.1\\models\\dict.txt"));
		sw = new StopWords("H:\\Coding Toolkits\\FudanNLP-1.6.1\\models\\stopwords");
		File idffile = new File(".\\selectedfeatures.txt");
		InputStreamReader isr = new InputStreamReader(new FileInputStream(idffile),"UTF-8");
		BufferedReader br = new BufferedReader(isr);
		String line;
		while((line = br.readLine())!=null)
		{
			String[] info = line.split("::::");
			//System.out.println(line);
			idf.put(info[1], Double.parseDouble(info[2]));
		}

		
	}
	public void adduserdictionary() throws IOException
	{
		File entityfile = new File(".\\militaryentities.txt");
		InputStreamReader isr = new InputStreamReader(new FileInputStream(entityfile),"UTF-8");
		BufferedReader br = new BufferedReader(isr);
		String line;
		while((line = br.readLine()) != null)
		{
			userdictionary.add(line);
		}
		
		br.close();
		isr.close();
		Dictionary dict = new Dictionary(false);
		dict.addSegDict(userdictionary);
		seg.setDictionary(dict);
	}
	
	public String ReadFile(String path) throws IOException
	{
		File file = new File(path);
		InputStreamReader isr = new InputStreamReader(new FileInputStream(file));
		BufferedReader br = new BufferedReader(isr);
		
		StringBuffer content = new StringBuffer();
		String line = null;
		while((line = br.readLine())!=null)
		{
		  content.append(line);	
		}
		br.close();
		isr.close();
		return content.toString();
	}
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub

		ExtractKeyWords EKWs = new ExtractKeyWords();
		AbstractExtractor key = new WordExtract(seg,sw);
		EKWs.adduserdictionary();
		
		//File file = new File("E:\\Project Archives\\�ٶȰٿ�\\����\\����\\�����۾��ˡ������Ƶ�����.txt");
        File file = new File("H:\\1.txt");
		String content  = EKWs.ReadFile(file.getAbsolutePath());

		byte[] contentbytes = content.getBytes("UTF-8");
		String[] words = seg.tag(new String(contentbytes,0,contentbytes.length,"UTF-8")).split("\\s+");
		HashMap<String,Integer> docwordstf = new HashMap<String,Integer>();
		for(String word : words)
		{
			if(docwordstf.containsKey(word))
				docwordstf.put(word, docwordstf.get(word)+1);
			else docwordstf.put(word, 1);
		}
		HashMap<String,Double> keywordweight = new HashMap<String,Double>();
		for(String word : docwordstf.keySet())
		{
			if(idf.containsKey(word))
				keywordweight.put(word, docwordstf.get(word)*idf.get(word));
		}
		ArrayList<Entry<String, Integer>> array = new ArrayList<Entry<String, Integer>>(
				docwordstf.entrySet());

		Collections.sort(array, new Comparator<Map.Entry<String, Integer>>() {
			public int compare(Map.Entry<String, Integer> o1,
					Map.Entry<String, Integer> o2) {
				if (o2.getValue() - o1.getValue() > 0)
					return 1;
				else if (o2.getValue() - o1.getValue() == 0)
					return 0;
				else
					return -1;
			}
		});
		
		
		
		System.out.println("tfidf��ȡ�ؼ���");
		for(int i=0;i<array.size();i++)
			if(userdictionary.contains(array.get(i).getKey())&&array.get(i).toString().length()>=6)
			System.out.println(array.get(i).getKey()+"\t"+array.get(i).getValue());
		
		//System.out.println(content);
		String str = "�ݿ�˹�工��p-32��p-46��p-50��p-80���������ϳ���������ǿ";//" F16ս����ý������о���������, �߼������ھ�(data mining)���ѡ� ��phone�������й����ӿƼ�����32�����������Ư������˿����";
		
		
		
		System.out.println(seg.tag(content));
		
		
/*		System.out.println("\n������ʱ�ʵ䣺");
		ArrayList<String> al = new ArrayList<String>();
		al.add("��Ѷ���Թܼ�");
		al.add("�ݿ�˹�工��p-32��p-46��p-50��p-80���������ϳ�");
		al.add("���������ϳ�");
		al.add("�й����ӿƼ�����32��");
		Dictionary dict = new Dictionary(false);
		dict.addSegDict(al);
		seg.setDictionary(dict);*/
		
		System.out.println(seg.tag(content));
		System.out.println("keyword api ��ȡ");
		Map<String,Integer> keywords = key.extract(content, 60);
		for(String k : keywords.keySet())
		  System.out.println(k+"  "+keywords.get(k));
		
	}

}
