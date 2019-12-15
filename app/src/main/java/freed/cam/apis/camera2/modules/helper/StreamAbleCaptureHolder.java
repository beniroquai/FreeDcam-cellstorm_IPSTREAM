package freed.cam.apis.camera2.modules.helper;

import android.hardware.camera2.CameraCharacteristics;
import android.media.Image;
import android.os.Build;
import android.support.annotation.RequiresApi;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.Socket;
import java.nio.ByteBuffer;

import freed.ActivityInterface;
import freed.cam.apis.basecamera.modules.ModuleInterface;
import freed.cam.apis.basecamera.modules.WorkFinishEvents;

public class StreamAbleCaptureHolder extends ImageCaptureHolder {

    //the connection to the server
    private Socket socket;
    //used to send data to the target
    private BufferedOutputStream bufferedOutputStream;

    public StreamAbleCaptureHolder(CameraCharacteristics characteristicss, boolean isRawCapture, boolean isJpgCapture, ActivityInterface activitiy, ModuleInterface imageSaver, WorkFinishEvents finish, RdyToSaveImg rdyToSaveImg, Socket socket) {
        super(characteristicss, isRawCapture, isJpgCapture, activitiy, imageSaver, finish, rdyToSaveImg);
        this.socket =socket;
        try {
            this.bufferedOutputStream = new BufferedOutputStream(socket.getOutputStream());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.KITKAT)
    @Override
    public void saveImage(Image image, String f) {
        ByteBuffer buffer = image.getPlanes()[0].getBuffer();
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes);
        try {
            //sending plain bayer bytearray with simple start end of file
            bufferedOutputStream.write("-1".getBytes());
            bufferedOutputStream.write(bytes);
            bufferedOutputStream.write("-2".getBytes());
            bufferedOutputStream.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     *
     * @param x_crop_pos start crop position in the pixel area
     * @param y_crop_pos start crop position int the pixel area
     * @param buf_width length of the buffer in pixel size
     * @param buf_height length of the buffer in pixel size
     * @param image to get cropped
     * @return
     */
    private byte[] cropByteArray(int x_crop_pos ,int y_crop_pos, int buf_width, int buf_height, Image image)
    {
        byte bytes[] = new byte[buf_width*buf_height*2];
        int bytepos = 0;
        ByteBuffer buffer = image.getPlanes()[0].getBuffer();
        int rowsize = buf_width*2;
        for (int y = y_crop_pos; y <  y_crop_pos +buf_height*2;y++)
        {
            for (int x = x_crop_pos; x < x_crop_pos +buf_width*2;x++)
            {
                bytes[bytepos++] = buffer.get(y*rowsize +x);
            }
        }
        return bytes;
    }
}
