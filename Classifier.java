import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Vector;
import java.util.Map.Entry;

import edu.fudan.ml.types.Dictionary;
import edu.fudan.nlp.cn.tag.CWSTagger;
import edu.fudan.util.exception.LoadModelException;


public class Classifier {

	/**
	 * @param 分类主程序
	 * Date：20131202
	 */
	private static ArrayList filelist = new ArrayList();
	private int directoryid = 0;//root 目录的id
	private static List categorylist = new LinkedList<Category>();//目录列表
	public HashMap<String,Integer> allwordsdf = new HashMap<String,Integer>();
	public HashMap<String,Integer> allwordstf = new HashMap<String,Integer>();
	private HashSet<String> allwords = new HashSet<String>();//记录所有的词
	private HashSet<String> stopwords = new HashSet<String>();//
	private int allwordsnum = 0;
	private int alldocsnum = 0;
	private static CWSTagger tag;
	private int dim;
	private int m;
	private HieSVM SVM;
	public Classifier(int ml, int diml) throws Exception
	{
		tag = new CWSTagger("H:\\Coding Toolkits\\FudanNLP-1.6.1\\models\\seg.m",new Dictionary("H:\\Coding Toolkits\\FudanNLP-1.6.1\\models\\dict.txt"));
        SVM = new HieSVM(ml,diml);
	    m = ml;
        dim = diml;

		System.out.println("正在读入停止词");
		File sw = new File(".\\stopwords.txt");
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(sw),"UTF-8"));
		String line;
		while((line = br.readLine())!=null)
		{
			stopwords.add(new String(line.trim().getBytes("UTF-8"),"UTF-8"));
			//System.out.println(new String(line.trim().getBytes("UTF-8"),"UTF-8"));
		}
		System.out.println("停止词数量："+stopwords.size());
		//Thread.sleep(50000);
		br.close();
	}
	
	public void rectifyClassifier(HashMap<DocNode,String> classifyresult)
	{//此函数用于矫正模型，矫正时候不需要对文档重新分词了，但是选择特征、构建文档向量、训练模型的过程都必不可少
	  for(DocNode doc : classifyresult.keySet())
	  {
		  for(int i = 0;i< categorylist.size();++i)
			{
				Category category = (Category)categorylist.get(i);
				if(doc.label == category.categoryid)
				{
					category.docList.add(doc);
					break;
				}		
			}
		  Category cat = new Category();
		  cat.bLeafCategory = true;
		  cat.categoryid = categorylist.size()+1;
		  cat.categoryname = classifyresult.get(doc);
		  cat.docList.add(doc);
		  categorylist.add(cat);
	  }
		
		
	}
	public String getCharset(String filePath) throws IOException{
        
        BufferedInputStream bin = new BufferedInputStream(new FileInputStream(filePath));  
        int p = (bin.read() << 8) + bin.read();  
        
        String code = null;  
        
        switch (p) {
            case 0xefbb:  
                code = "UTF-8";  
                break;  
            case 0xfffe:  
                code = "Unicode";  
                break;  
            case 0xfeff:  
                code = "UTF-16BE";  
                break;  
            default:  
                code = "GBK";
        }
        return code;
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
	public void refreshFileList(String strPath,Category cat) {
        File dir = new File(strPath); 
        File[] files = dir.listFiles(); 
        
        cat = new Category();
    	cat.absolutepath = strPath;
    	cat.categoryid = directoryid-1;
    	cat.categoryname = dir.getName();
    	
        if (files == null)
            return;
        for (int i = 0; i < files.length; i++) {
            if (files[i].isDirectory()) {
                directoryid++;
                refreshFileList(files[i].getAbsolutePath(),cat);
                File subdir = new File(files[i].getAbsolutePath());
                File[] subfiles = subdir.listFiles();
                int j =0;
                for( j = 0; j < subfiles.length; j++)
                {
                	if(subfiles[j].isDirectory()) break;
                }
                if(j == subfiles.length)
                	System.out.println("Leaf category = "+files[i].getAbsolutePath()+" id = "+(directoryid-1));//得到此文件夹为叶子目录，需要记录下来
            } else {
                String strFileName = files[i].getAbsolutePath();
                //System.out.println("---"+strFileName);
                //filelist.add(files[i].getAbsolutePath());
                DocNode doc = new DocNode(directoryid-1,strFileName);
                cat.docList.add(doc);
                cat.bLeafCategory = true;
            }//end of if
        }//end of for
        categorylist.add(cat);
        
    }
	public void generateDocumentVector(Vector<DocNode> DocList) throws NumberFormatException, IOException
	{//此函数用于生成待分类文档的特征向量
		HashMap<String,Double> featuresidf = new HashMap<String,Double>();//训练模型挑选的特征及其tfidf值
		HashMap<String,Integer> featuresindex = new HashMap<String,Integer>();//训练模型挑选的特征及其index值
		
		File selectedfeatures = new File(".\\selectedfeatures.txt");
		if(!selectedfeatures.exists()) 
		{
			System.out.println("特征文件\"selectedfeatures.txt\"不存在！请训练分类模型");
			return;
		}
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(selectedfeatures),"UTF-8"));
		String line;
		while((line = br.readLine())!=null)
		{
			String[] info = line.split("::::");
			featuresidf.put(info[1],Double.parseDouble(info[2]));
			featuresindex.put(info[1], Integer.parseInt(info[0]));
		}
		System.out.println("训练模型挑选的特征数量："+featuresidf.size());
		br.close();
		
		File unclassifyfilefeaturevector = new File(".\\unclassifyfilefeaturevector.txt");
		if(unclassifyfilefeaturevector.exists()) unclassifyfilefeaturevector.delete();
		unclassifyfilefeaturevector.createNewFile();
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(unclassifyfilefeaturevector),"UTF-8"));
		
		for(DocNode doc : DocList)
		{
			for(String feature : featuresidf.keySet())
			{
				if(doc.tf.containsKey(feature))
				{
					bw.write(featuresindex.get(feature)+":"+featuresidf.get(feature)*doc.tf.get(feature)+" ");
					bw.flush();
				}
			}
			bw.write("\r\n");bw.flush();
		}//end of for(DocNode doc:
		System.out.println("生成文档向量过程结束");
		
	}
	public void generateFeatureVector(int mode,boolean bWordSeg) throws UnsupportedEncodingException, IOException
	{//mode 对应三种模式1-->Train;2-->Test;
		allwordsdf.clear();
		allwordstf.clear();
		allwords.clear();
		allwordsnum = 0;
		alldocsnum = 0;
		System.out.println("========== 开始分词  ===========");
        Category cat;
		for(int i=0;i<categorylist.size();++i)
		{
			cat = (Category)(categorylist.get(i));
			if(cat.bLeafCategory == true)
			{
				System.out.println("分词目录："+cat.categoryname);
				Vector<DocNode> docList = cat.docList;
				for(DocNode doc : docList)
				{
					alldocsnum++;
					//System.out.println(doc.absolutepath);
					String content = ReadFile(doc.absolutepath);
					String[] words;
					if(bWordSeg )//是否已分词
					{
						words = content.split("\\s+");
					}else
					{
						//byte nativeBytes[] = nlpir.NLPIR_ParagraphProcess(content.getBytes("UTF-8"), 0);
						//words = (new String(nativeBytes, 0, nativeBytes.length, "UTF-8")).split(" +");
						byte[] contentbytes = content.getBytes("UTF-8");
						words = tag.tag(new String(contentbytes,0,contentbytes.length,"UTF-8")).split("\\s+");
					}
					
					HashSet<String> docwords = new HashSet<String>();
					
					for(String word: words)
					{//计算每一个文档的tf向量
						if(!stopwords.contains(word)&&word.trim().length()>0)
						{
							if(doc.tf.containsKey(word))
								doc.tf.put(word, doc.tf.get(word)+1);
							else doc.tf.put(word, 1);
							
							allwords.add(word);
							
							if(allwordstf.containsKey(word))
								allwordstf.put(word, allwordstf.get(word)+1);
							else allwordstf.put(word, 1);
							docwords.add(word);
						}
					}
					for(String word:docwords)
					{//记录每个word的df值
						if(allwordsdf.containsKey(word))
							allwordsdf.put(word, allwordsdf.get(word)+1);
						else allwordsdf.put(word, 1);
					}
				}//end of for(DocNode doc
			}
			//System.out.println(cat.categoryname+"  "+cat.absolutepath+"  "+cat.categoryid+"  "+cat.bLeafCategory+" "+cat.fileList.size());
		}//end of for(int i=0;i<categorylist.size();++i)
		allwordsnum = allwords.size();
		System.out.println("总词数目："+allwordsnum);
		System.out.println("总文档数目："+alldocsnum);
			
		System.out.println("分词结束");
		HashMap<String,Integer> features = new HashMap<String,Integer>();//挑选的特征
		if(1 == mode)//此时为训练模式
		{
			System.out.println("开始挑选特征");
			HashMap<String,Double> tfidf = new HashMap<String,Double>();
			for(String word:allwords)
			{
				tfidf.put(word, (double)allwordstf.get(word)/allwordsnum * Math.log((double)alldocsnum/allwordsdf.get(word)));
			}
			
			//更具词的tfidf排序挑选特征，并记录其idf值
			ArrayList<Entry<String, Double>> array = new ArrayList<Entry<String, Double>>(
					tfidf.entrySet());

			Collections.sort(array, new Comparator<Map.Entry<String, Double>>() {
				public int compare(Map.Entry<String, Double> o1,
						Map.Entry<String, Double> o2) {
					if (o2.getValue() - o1.getValue() > 0)
						return 1;
					else if (o2.getValue() - o1.getValue() == 0)
						return 0;
					else
						return -1;
				}
			});

			File selectedfeatures = new File(".\\selectedfeatures.txt");
			if(selectedfeatures.exists()) selectedfeatures.delete();
			selectedfeatures.createNewFile();
			BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(selectedfeatures),"UTF-8"));
			
			for (int i = 0;i< dim;i++) {
				System.out.println(array.get(i).getKey() + ":" + array.get(i).getValue());
				features.put(array.get(i).getKey(), i);
				bw.write(i+"::::"+array.get(i).getKey()+"::::"+Math.log((double)alldocsnum/allwordsdf.get(array.get(i).getKey()))+"\r\n");
				bw.flush();
			}
			bw.close();
			System.out.println("挑选特征结束");
		}//end of if(Train)
		else if(2 == mode)//测试模式
		{
			File selectedfeatures = new File(".\\selectedfeatures.txt");
			BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(selectedfeatures),"UTF-8"));
			String line;
			while((line = br.readLine())!=null)
			{
				String[] info = line.split("::::");
				features.put(info[1],Integer.parseInt(info[0]));
			}
			System.out.println("挑选的特征数量："+features.size());
			br.close();
		}//end of Test
		System.out.println("开始生成文档向量");
		File featurevectorfile = new File("");
		
		if(1 == mode) featurevectorfile = new File(".\\trainfeaturevectorfile.txt");
		else if(2 == mode) featurevectorfile = new File(".\\testfeaturevectorfile.txt");
		
		if(featurevectorfile.exists()) featurevectorfile.delete();
		featurevectorfile.createNewFile();
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(featurevectorfile),"UTF-8"));
		for(int i=0;i<categorylist.size();++i)
		{			
			cat = (Category)(categorylist.get(i));
			if(cat.bLeafCategory == true)
			{
				Vector<DocNode> docList = cat.docList;
				for(DocNode doc : docList)
				{
					//System.out.println("Scanning document:"+doc.absolutepath);
					//bw.write(doc.label+" ");bw.flush();
					for(String feature:features.keySet())
					{
						if(doc.tf.containsKey(feature))
						{
							//bw.write(index+":"+(double)doc.tf.get(feature)/doc.tf.size()*Math.log10(alldocsnum/(double)allwordsdf.get(feature))+" ");
							bw.write(features.get(feature)+":"+(double)doc.tf.get(feature)/doc.tf.size()*Math.log(alldocsnum/(double)allwordsdf.get(feature))+" ");
							bw.flush();
						}/*else
						{
							bw.write(features.get(feature)+":"+0+" ");
							bw.flush();
						}*/
					}
					bw.write(doc.label+" ");bw.flush();
					bw.write("\r\n");bw.flush();
				}//end of for(DocNode doc:
			}//end of if
		}//end of for(int i
	}
	public void train(String root, boolean bWordSeg) throws Exception
	{
		Category cat = new Category();
		directoryid = 0;
    	cat.absolutepath = root;
    	cat.categoryid = -1;
    	cat.categoryname = (new File(root)).getName();
		refreshFileList(root,cat);
		File categoryinfo = new File(".\\categoryinfo.txt");
		if(categoryinfo.exists()) categoryinfo.delete();
		categoryinfo.createNewFile();
		
		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(categoryinfo),"UTF-8"));
		
		for(int i = 0; i < categorylist.size(); ++i)
		{
		 	Category category = (Category) categorylist.get(i);
		 	bw.write(category.categoryid+"::::"+category.categoryname+"\r\n");
		 	bw.flush();
		}
		bw.close();
		
		System.out.println("===========训练模型准备工作开始==============");
        generateFeatureVector(1,bWordSeg);//"Train"模式
		
        System.out.println("........SVM训练运行开始..........");
        
        String train_file = ".\\trainfeaturevectorfile.txt";
        
        HieSVM SVM = new HieSVM(m, dim);
        BufferedReader br=new BufferedReader(new InputStreamReader(new FileInputStream(train_file)));
        String temp=br.readLine();
        Vector trainfiles = new Vector();//训练文件的特征向量
        while(temp!=null){//读取训练样本向量表
        	trainfiles.add(temp);
        	temp=br.readLine();
        }
		br.close();
		Vector testfiles = new Vector();
		int deletedcount = 0;
		System.out.println("original trainfiles size = "+trainfiles.size());
		for(int i=0; i<trainfiles.size();++i)
			if(i%3 == 0)
			{
				testfiles.add(trainfiles.elementAt(i-deletedcount));
				trainfiles.removeElementAt(i-deletedcount);
				deletedcount++;
			}
		System.out.println("final trainfiles size = "+trainfiles.size());
		
        SVM.RDA_large(trainfiles, testfiles, 1, 0.0001);
        System.out.println("........SVM训练结束..........");
	}
	public void test(String root, boolean bWordSeg) throws Exception
	{
		Category cat = new Category();
		directoryid=0;
    	cat.absolutepath = root;
    	cat.categoryid = -1;
    	cat.categoryname = (new File(root)).getName();
    	categorylist.clear();
    	System.out.println("扫描待分类文档目录");
		refreshFileList(root,cat);
		generateFeatureVector(2,bWordSeg);//"Test"模式
		System.out.println("正在进行分类测试");
        
		String test_file = ".\\testfeaturevectorfile.txt";
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(test_file)));
        String temp=br.readLine();
        Vector testfiles=new Vector();//测试文件的特征向量
        
        while(temp!=null){//读取测试样本向量表
        	testfiles.add(temp);
        	temp=br.readLine();
        }
		br.close();
		HieSVM SVM = new HieSVM(m, dim);
		SVM.ReadNormalVector(m, dim);
		SVM.test_classify_sparse(testfiles,1);//在test模式下
		System.out.println("分类模型测试结束");
	}
	public void classify(String dirpath, boolean bWordSeg) throws Exception
	{	
		File filesroot = new File(dirpath);
		File[] files = filesroot.listFiles();
		Vector<DocNode> DocList = new Vector<DocNode>();
		for(File f : files)
		{
		  DocNode doc = new DocNode(-1,f.getAbsolutePath());
		  String content = ReadFile(f.getAbsolutePath());
		  String words[];
		  if(bWordSeg)//是否已分词
		  {
			  words = content.split(" +");
		  }else{
			  //byte nativeBytes[] = nlpir.NLPIR_ParagraphProcess(content.getBytes("UTF-8"), 0);
			  //words = (new String(nativeBytes, 0, nativeBytes.length, "UTF-8")).split(" +");
			  byte[] contentbytes = content.getBytes("UTF-8");
			  words = tag.tag(new String(contentbytes,0,contentbytes.length,"UTF-8")).split("\\s+");			
		  }

		  for(String word : words)
		  {
			  if(doc.tf.containsKey(word))
				  doc.tf.put(word, doc.tf.get(word)+1);
			  else doc.tf.put(word, 1);
		  }
		  DocList.add(doc);
		}
		
		generateDocumentVector(DocList);//"Classification"模式
		
		File unclassifyfilefeaturevector = new File(".\\unclassifyfilefeaturevector.txt");
		if(!unclassifyfilefeaturevector.exists())
		 {
			System.out.println("待分类文档向量文件不存在");
			return;
		 }
		Vector unclassifyfiles = new Vector<String>();
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(unclassifyfilefeaturevector),"UTF-8"));
		String line;
		int count = 0;
		while((line = br.readLine())!=null)
		{
			unclassifyfiles.add(line);
		}
		br.close();
		SVM.ReadNormalVector(SVM.m, SVM.dim);//读入已经训练好的SVM分类器参数
		Vector<String> classifyresult = new Vector<String>();
		SVM.classifyfiles(unclassifyfiles, classifyresult);
		//生成的结果classifyresult可以显式在网页上
	}
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
         
		int ml = 28;
		int diml = 10000;
		boolean bWordSeg = false;
		Classifier classifier = new Classifier(ml, diml);//参数为模型特征向量的维度
		/*Category cat = new Category();
		classifier.refreshFileList("D:\\Documents and Settings\\dujintao\\桌面\\训练测试集\\两层训练及测试集\\train两级",cat);
		for(int i = 0; i < categorylist.size(); ++i)
		{
		 	Category category = (Category) categorylist.get(i);
		 	System.out.println(category.categoryid+"::::"+category.categoryname);
		 	
		}
		Thread.sleep(50000);*/

        //String root = "E:\\Project Archives\\文本分类系统\\文本分类系统[测试预料]\\测试语料\\TanCorpHier-Txt";
		//String train_root = "E:\\Project Archives\\文本分类系统\\文本分类系统[测试语料]\\测试语料\\fdcorpus_15\\15classes_train";
		//String train_root = "D:\\Documents and Settings\\dujintao\\桌面\\训练测试集\\两层训练及测试集\\train两级";
		String train_root = "E:\\Project Archives\\内容管理系统\\百度百科\\分类";
		//String test_root = "E:\\Project Archives\\文本分类系统\\文本分类系统[测试语料]\\测试语料\\fdcorpus_15\\15classes_test";
		String test_root = "D:\\Documents and Settings\\dujintao\\桌面\\训练测试集\\两层训练及测试集\\test两级";
		classifier.train(train_root,bWordSeg);
		//classifier.test(test_root,bWordSeg);
		//String classifydir = "D:\\Documents and Settings\\dujintao\\桌面\\训练测试集\\两层训练及测试集\\test两级\\体育\\2篮球";
		//String classifydir = "E:\\Project Archives\\文本分类系统\\文本分类系统[测试语料]\\测试语料\\fdcorpus_15\\15classes_test\\C5-EDUCATION";
		String classifydir = "E:\\Project Archives\\内容管理系统\\百度百科\\分类\\导弹";
		//classifier.classify(classifydir, bWordSeg);
	   
	}
}
