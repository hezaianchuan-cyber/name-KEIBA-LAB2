package jp.keibalab.app;

import android.app.Activity;
import android.os.Bundle;
import android.webkit.*;
import android.content.Intent;
import android.net.Uri;
import android.util.Base64;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends Activity {
    private WebView webView;
    private ValueCallback<Uri[]> fileCallback;
    private static final int FILE_REQUEST = 701;
    private final ExecutorService downloadExecutor = Executors.newSingleThreadExecutor();

    @Override public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        webView = new WebView(this);
        webView.setBackgroundColor(0xFF101A2B);
        WebSettings s = webView.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setAllowFileAccess(true);
        s.setAllowContentAccess(true);
        s.setDefaultTextEncodingName("UTF-8");
        webView.addJavascriptInterface(new OfficialDataBridge(), "AndroidBridge");
        webView.setWebChromeClient(new WebChromeClient() {
            @Override public boolean onShowFileChooser(WebView view, ValueCallback<Uri[]> callback, FileChooserParams params) {
                if (fileCallback != null) fileCallback.onReceiveValue(null);
                fileCallback = callback;
                Intent intent = params.createIntent();
                try { startActivityForResult(intent, FILE_REQUEST); }
                catch (Exception e) { fileCallback = null; return false; }
                return true;
            }
        });
        webView.setWebViewClient(new WebViewClient());
        setContentView(webView);
        webView.loadUrl("file:///android_asset/index.html");
    }

    private class OfficialDataBridge {
        @JavascriptInterface
        public void downloadDailyData() {
            downloadExecutor.execute(() -> {
                JSONObject response = new JSONObject();
                JSONArray files = new JSONArray();
                JSONArray errors = new JSONArray();
                try {
                    addDailyFile(files, errors, "レース情報",
                        "https://www.keiba.go.jp/KeibaWeb/DataDownload/RaceDataDownload?type=daily");
                    addDailyFile(files, errors, "オッズ情報",
                        "https://www.keiba.go.jp/KeibaWeb/DataDownload/OddsDataDownload?type=daily");
                    response.put("files", files);
                    response.put("errors", errors);
                    if (files.length() == 0) response.put("error", errors.length() > 0 ? errors.getString(0) : "公式データを取得できませんでした");
                } catch (Exception e) {
                    try { response.put("error", e.getMessage() == null ? "取得処理に失敗しました" : e.getMessage()); }
                    catch (Exception ignored) {}
                }
                final String payload = response.toString();
                runOnUiThread(() -> {
                    if (webView != null) {
                        webView.evaluateJavascript("window.onOfficialDataReceived(" + JSONObject.quote(payload) + ")", null);
                    }
                });
            });
        }
    }

    private void addDailyFile(JSONArray files, JSONArray errors, String label, String address) {
        try {
            byte[] data = downloadZip(address);
            JSONObject item = new JSONObject();
            item.put("label", label);
            item.put("base64", Base64.encodeToString(data, Base64.NO_WRAP));
            files.put(item);
        } catch (Exception e) {
            try { errors.put(label + ": " + (e.getMessage() == null ? "取得失敗" : e.getMessage())); }
            catch (Exception ignored) {}
        }
    }

    private byte[] downloadZip(String address) throws Exception {
        HttpURLConnection connection = (HttpURLConnection) new URL(address).openConnection();
        connection.setRequestMethod("GET");
        connection.setConnectTimeout(20000);
        connection.setReadTimeout(60000);
        connection.setInstanceFollowRedirects(true);
        connection.setRequestProperty("User-Agent", "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 Chrome/120 Mobile Safari/537.36");
        connection.setRequestProperty("Accept", "application/zip, application/octet-stream, */*");
        connection.setRequestProperty("Referer", "https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/TodayRaceInfoTop");
        try {
            int status = connection.getResponseCode();
            if (status != HttpURLConnection.HTTP_OK) throw new Exception("HTTP " + status);
            try (InputStream input = connection.getInputStream();
                 ByteArrayOutputStream output = new ByteArrayOutputStream()) {
                byte[] buffer = new byte[8192];
                int read;
                while ((read = input.read(buffer)) != -1) {
                    output.write(buffer, 0, read);
                    if (output.size() > 25 * 1024 * 1024) throw new Exception("ZIPが25MBを超えています");
                }
                byte[] bytes = output.toByteArray();
                if (bytes.length < 4 || bytes[0] != 'P' || bytes[1] != 'K') throw new Exception("ZIP形式のデータではありません");
                return bytes;
            }
        } finally {
            connection.disconnect();
        }
    }

    @Override protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == FILE_REQUEST && fileCallback != null) {
            Uri[] result = WebChromeClient.FileChooserParams.parseResult(resultCode, data);
            fileCallback.onReceiveValue(result);
            fileCallback = null;
        }
    }

    @Override protected void onDestroy() {
        downloadExecutor.shutdownNow();
        super.onDestroy();
    }

    @Override public void onBackPressed() {
        if (webView != null && webView.canGoBack()) webView.goBack();
        else super.onBackPressed();
    }
}
