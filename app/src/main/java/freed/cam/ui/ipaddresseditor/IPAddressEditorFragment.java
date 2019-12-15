/*
 *
 *     Copyright (C) 2015 Ingo Fuchs
 *     This program is free software; you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation; either version 2 of the License, or
 *     (at your option) any later version.
 *
 *     This program is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 *
 *     You should have received a copy of the GNU General Public License along
 *     with this program; if not, write to the Free Software Foundation, Inc.,
 *     51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 * /
 */

package freed.cam.ui.ipaddresseditor;

import android.app.AlertDialog.Builder;
import android.content.DialogInterface;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.support.annotation.Nullable;
import android.support.v4.app.Fragment;
import android.view.LayoutInflater;
import android.view.MenuItem;
import android.view.View;
import android.view.View.OnClickListener;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.PopupMenu;
import android.widget.PopupMenu.OnMenuItemClickListener;
import android.widget.Switch;
import android.widget.Toast;

import com.troop.freedcam.R.id;
import com.troop.freedcam.R.layout;

import java.util.HashMap;

import freed.settings.SettingKeys;
import freed.settings.SettingsManager;
import freed.utils.VideoMediaProfile;
import freed.utils.VideoMediaProfile.VideoMode;

/**
 * Created by troop on 15.02.2016.
 */
public class IPAddressEditorFragment extends Fragment {
    final String TAG = IPAddressEditorFragment.class.getSimpleName();

    private EditText editText_ipaddress;
    private EditText editText_ipaddress_port;
    private VideoMediaProfile currentProfile;

    private String mIPAddress = "192.168.43.86";//"192.168.43.83";
    private int mPort = 1234;

    private HashMap<String, VideoMediaProfile> videoMediaProfiles;
    @Nullable
    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState)
    {
        return inflater.inflate(layout.ip_address_editor_fragment,container,false);
    }

    @Override
    public void onViewCreated(View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        editText_ipaddress = view.findViewById(id.editText_ProfileName);
        editText_ipaddress_port = view.findViewById(id.editText_Profilewidth);

        Button button_save = view.findViewById(id.button_Save_profile);
        button_save.setOnClickListener(onSavebuttonClick);

    }


    private void setIPAddressPort(String myIPAddress, int myPort) {
        mIPAddress = myIPAddress;
        mPort = myPort;

    }

    private final OnClickListener onSavebuttonClick = new OnClickListener() {
        @Override
        public void onClick(View v)
        {
            mIPAddress = String.valueOf(editText_ipaddress.getText());
            mPort = Integer.parseInt(String.valueOf(editText_ipaddress_port.getText()));
            setIPAddressPort(mIPAddress, mPort);
            Toast.makeText(getContext(),"IP Address Set", Toast.LENGTH_SHORT).show();
        }
    };
}
